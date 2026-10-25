import copy
import logging
import re
from bisect import bisect_left, bisect_right
from functools import lru_cache
from typing import Iterator, List, Optional, Tuple

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.context_aware_enhancers import ContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpArtifacts

logger = logging.getLogger("presidio-analyzer")


# Phrase words are split at whitespace, hyphens and slashes, and between two
# words of the text any whitespace, or one hyphen or slash, joins them:
# "driving licence" also matches "driving-licence", and "zip code" "zip/code".
_PHRASE_WORD_SEPARATOR = re.compile(r"[\s\-/]+")
_PHRASE_GAP = r"(?:\s+|\s*[-/]\s*)"
_PHRASE_GAP_RE = re.compile(_PHRASE_GAP)
_WORD_CHAR = re.compile(r"\w")


@lru_cache(maxsize=1024)
def _phrase_words(phrase: str) -> Tuple[str, ...]:
    """Split a context phrase into its lowercase words."""
    return tuple(word for word in _PHRASE_WORD_SEPARATOR.split(phrase.lower()) if word)


@lru_cache(maxsize=1024)
def _compile_phrase(phrase: str, whole_word: bool) -> re.Pattern:
    """Compile a context phrase: its words in order, joined by a phrase gap.

    The phrase starts at a word boundary, so "st-nr" does not match "first nr";
    only "whole_word" also requires a word boundary after it.
    """
    pattern = _PHRASE_GAP.join(re.escape(word) for word in _phrase_words(phrase))
    pattern = rf"(?<!\w){pattern}"
    if whole_word:
        pattern = rf"{pattern}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


class LemmaContextAwareEnhancer(ContextAwareEnhancer):
    """
    A class representing a lemma based context aware enhancer logic.

    Context words might enhance confidence score of a recognized entity,
    LemmaContextAwareEnhancer is an implementation of Lemma based context aware logic,
    it compares spacy lemmas of each word in context of the matched entity to given
    context and the recognizer context words,
    if matched it enhance the recognized entity confidence score by a given factor.
    A context word that contains whitespace (e.g. 'driving licence') cannot equal
    a single lemma, so when no context word matches, such phrases are also looked
    up in the original text of the context window, outside the entity itself.
    For an entity of several tokens, when no context word matches, the
    `context_suffix_count` keywords after its last token also count, so the
    entity's own tokens do not use up the suffix window.

    :param context_similarity_factor: How much to enhance confidence of match entity
    :param min_score_with_context_similarity: Minimum confidence score
    :param context_prefix_count: how many words before the entity to match context
    :param context_suffix_count: how many words after the entity to match context
    :param context_matching_mode: Matching mode for context words. Options:
        - "substring" (default): Match context words as substrings
          (e.g., 'card' matches 'creditcard', 'lic' matches 'duplicate').
          Maintains backward compatibility.
        - "whole_word": Match context words only as whole words
          (e.g., 'lic' matches 'lic' but not 'duplicate').
          Prevents false positives.
        A phrase matches its words in order, case-insensitively, separated by
        whitespace or by one hyphen or slash, starting at a word boundary.
        "whole_word" also requires a word boundary after it, and also accepts
        consecutive tokens whose text or lemma equals each word
        ('driving licence' matches 'driving licences').
    """

    def __init__(
        self,
        context_similarity_factor: float = 0.35,
        min_score_with_context_similarity: float = 0.4,
        context_prefix_count: int = 5,
        context_suffix_count: int = 0,
        context_matching_mode: str = "substring",
    ):
        super().__init__(
            context_similarity_factor=context_similarity_factor,
            min_score_with_context_similarity=min_score_with_context_similarity,
            context_prefix_count=context_prefix_count,
            context_suffix_count=context_suffix_count,
        )
        if context_matching_mode not in ["whole_word", "substring"]:
            raise ValueError(
                f"context_matching_mode must be one of: 'whole_word', 'substring'. "
                f"Got: {context_matching_mode}"
            )
        self.context_matching_mode = context_matching_mode

    def enhance_using_context(
        self,
        text: str,
        raw_results: List[RecognizerResult],
        nlp_artifacts: NlpArtifacts,
        recognizers: List[EntityRecognizer],
        context: Optional[List[str]] = None,
    ) -> List[RecognizerResult]:
        """
        Update results in case the lemmas of surrounding words or input context
        words are identical to the context words.

        Using the surrounding words of the actual word matches, look
        for specific strings that if found contribute to the score
        of the result, improving the confidence that the match is
        indeed of that PII entity type

        :param text: The actual text that was analyzed
        :param raw_results: Recognizer results which didn't take
                            context into consideration
        :param nlp_artifacts: The nlp artifacts contains elements
                              such as lemmatized tokens for better
                              accuracy of the context enhancement process
        :param recognizers: the list of recognizers
        :param context: list of context words
        """  # noqa: D205,D400

        # create a deep copy of the results object, so we can manipulate it
        results = copy.deepcopy(raw_results)

        # create recognizer context dictionary
        recognizers_dict = {recognizer.id: recognizer for recognizer in recognizers}
        # context words of several words, which no single lemma can equal
        phrases_dict = {
            recognizer.id: [
                word
                for word in recognizer.context or []
                if len(_phrase_words(word)) > 1
            ]
            for recognizer in recognizers
        }

        # Create empty list in None or lowercase all context words in the list
        if not context:
            context = []
        else:
            context = [word.lower() for word in context]

        # Sanity
        if nlp_artifacts is None:
            logger.warning("NLP artifacts were not provided")
            return results

        for result in results:
            recognizer = None
            # get recognizer matching the result, if found.
            if (
                result.recognition_metadata
                and RecognizerResult.RECOGNIZER_IDENTIFIER_KEY
                in result.recognition_metadata.keys()
            ):
                recognizer = recognizers_dict.get(
                    result.recognition_metadata[
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY
                    ]
                )

            if not recognizer:
                logger.debug(
                    "Recognizer name not found as part of the "
                    "recognition_metadata dict in the RecognizerResult. "
                )
                continue

            # skip recognizer result if the recognizer doesn't support
            # context enhancement
            if not recognizer.context:
                logger.debug(
                    "recognizer '%s' does not support context enhancement",
                    recognizer.name,
                )
                continue

            # skip context enhancement if already boosted by recognizer level
            if result.recognition_metadata.get(
                RecognizerResult.IS_SCORE_ENHANCED_BY_CONTEXT_KEY
            ):
                logger.debug("result score already boosted, skipping")
                continue

            # extract lemmatized context from the surrounding of the match
            word = text[result.start : result.end]

            surrounding_words = self._extract_surrounding_words(
                nlp_artifacts=nlp_artifacts, word=word, start=result.start
            )

            # combine other sources of context with surrounding words
            surrounding_words.extend(context)

            supportive_context_word = self._find_supportive_word_in_context(
                surrounding_words, recognizer.context, self.context_matching_mode
            )
            phrases = phrases_dict[recognizer.id]
            if (
                supportive_context_word == ""
                and nlp_artifacts.tokens
                and (phrases or self.context_suffix_count > 0)
            ):
                first, last = self._find_entity_token_range(
                    nlp_artifacts, result.start, result.end
                )
                supportive_context_word = self._find_supportive_word_in_context(
                    self._extract_words_after_entity(nlp_artifacts, first, last),
                    recognizer.context,
                    self.context_matching_mode,
                )
                if supportive_context_word == "" and phrases:
                    supportive_context_word = self._find_supportive_phrase_in_window(
                        text, nlp_artifacts, first, last, result, phrases
                    )
            if supportive_context_word != "":
                result.score += self.context_similarity_factor
                result.score = max(result.score, self.min_score_with_context_similarity)
                result.score = min(result.score, ContextAwareEnhancer.MAX_SCORE)

                # Update the explainability object with context information
                # helped to improve the score
                result.analysis_explanation.set_supportive_context_word(
                    supportive_context_word
                )
                result.analysis_explanation.set_improved_score(result.score)
        return results

    @staticmethod
    def _find_supportive_word_in_context(
        context_list: List[str],
        recognizer_context_list: List[str],
        matching_mode: str = "substring",
    ) -> str:
        """
        Find words in the text which are relevant for context evaluation.

        A word is considered a supportive context word based on the matching mode:
        - "substring" (default): Substring match
          (e.g., 'card' matches 'creditcard', 'lic' matches 'duplicate')
        - "whole_word": Exact whole-word match (case-insensitive)
          (e.g., 'lic' matches 'lic' but not 'duplicate')

        :param context_list words before and after the matched entity within
               a specified window size
        :param recognizer_context_list a list of words considered as
                context keywords manually specified by the recognizer's author
        :param matching_mode: Matching mode ('whole_word' or 'substring').
               Defaults to 'substring'.
        """
        word = ""
        # If the context list is empty, no need to continue
        if context_list is None or recognizer_context_list is None:
            return word

        for predefined_context_word in recognizer_context_list:
            result = False

            if matching_mode == "substring":
                # Substring match (case-insensitive) - default behavior
                # for backward compatibility
                result = next(
                    (
                        True
                        for keyword in context_list
                        if predefined_context_word.lower() in keyword.lower()
                    ),
                    False,
                )
            elif matching_mode == "whole_word":
                # Exact whole-word match (case-insensitive)
                result = next(
                    (
                        True
                        for keyword in context_list
                        if predefined_context_word.lower() == keyword.lower()
                    ),
                    False,
                )

            if result:
                logger.debug("Found context keyword '%s'", predefined_context_word)
                word = predefined_context_word
                break

        return word

    def _find_supportive_phrase_in_window(
        self,
        text: str,
        nlp_artifacts: NlpArtifacts,
        first: int,
        last: int,
        result: RecognizerResult,
        phrases: List[str],
    ) -> str:
        """
        Find a context word of several words in the context window.

        Such a phrase never equals a single lemma, so it is matched against the
        original text of the window, and in "whole_word" mode also against the
        text or lemma of its consecutive tokens. A match that overlaps the
        entity itself does not count.

        :param text: the analyzed text
        :param nlp_artifacts: the NLP artifacts of the analyzed text
        :param first: index of the entity's first token
        :param last: index of the entity's last token
        :param result: the recognized entity
        :param phrases: the recognizer's context words of several words
        :return: the first matching phrase, or an empty string
        """
        window = self._get_context_window_tokens(nlp_artifacts, first, last)
        if not window:
            return ""
        tokens = nlp_artifacts.tokens
        tokens_indices = nlp_artifacts.tokens_indices
        window_start = tokens_indices[window[0]]
        window_end = tokens_indices[window[1]] + len(tokens[window[1]])
        whole_word = self.context_matching_mode == "whole_word"
        window_words = None
        for phrase in phrases:
            if self._phrase_in_window_text(
                text, window_start, window_end, result, phrase, whole_word
            ):
                return phrase
            if whole_word:
                if window_words is None:
                    # (index, lowercase text, lowercase lemma) of each token,
                    # without the whitespace, hyphen and slash tokens a phrase
                    # gap holds
                    window_words = [
                        (i, str(tokens[i]).lower(), nlp_artifacts.lemmas[i].lower())
                        for i in range(window[0], window[1] + 1)
                        if str(tokens[i]) not in ("-", "/")
                        and not str(tokens[i]).isspace()
                    ]
                if self._phrase_in_window_tokens(
                    text, nlp_artifacts, window_words, result, phrase
                ):
                    return phrase
        return ""

    @staticmethod
    def _phrase_in_window_text(
        text: str,
        window_start: int,
        window_end: int,
        result: RecognizerResult,
        phrase: str,
        whole_word: bool,
    ) -> bool:
        """Check if a phrase occurs in the original text of the context window.

        :param text: the analyzed text
        :param window_start: start offset of the context window in the text
        :param window_end: end offset of the context window in the text
        :param result: the recognized entity
        :param phrase: a context word of several words
        :param whole_word: whether the phrase needs word boundaries around it
        """
        pattern = _compile_phrase(phrase, whole_word)
        # One character past the window lets a word boundary see the next
        # character; a match ending there is outside the window. Search again
        # one character on, as a rejected match can overlap a valid one.
        end_pos = window_end + 1
        match = pattern.search(text, window_start, end_pos)
        while match:
            if match.end() <= window_end and (
                match.end() <= result.start or match.start() >= result.end
            ):
                logger.debug("Found context phrase '%s'", phrase)
                return True
            match = pattern.search(text, match.start() + 1, end_pos)
        return False

    @staticmethod
    def _phrase_in_window_tokens(
        text: str,
        nlp_artifacts: NlpArtifacts,
        window_words: List[Tuple[int, str, str]],
        result: RecognizerResult,
        phrase: str,
    ) -> bool:
        """Check if consecutive window tokens spell the phrase by text or lemma.

        Each phrase word must equal the text or the lemma of its token (as a
        single context word equals a lemma), and only a phrase gap may lie
        between two tokens. The tokens must not continue a word of the text on
        either side, and must not overlap the entity.

        :param text: the analyzed text
        :param nlp_artifacts: the NLP artifacts of the analyzed text
        :param window_words: index, lowercase text and lowercase lemma of each
            window token that is not a hyphen or slash
        :param result: the recognized entity
        :param phrase: a context word of several words
        """
        words = _phrase_words(phrase)
        tokens = nlp_artifacts.tokens
        tokens_indices = nlp_artifacts.tokens_indices
        for j in range(len(window_words) - len(words) + 1):
            run = window_words[j : j + len(words)]
            if not all(word in (t, lemma) for word, (_, t, lemma) in zip(words, run)):
                continue
            first, last = run[0][0], run[-1][0]
            run_start = tokens_indices[first]
            run_end = tokens_indices[last] + len(tokens[last])
            if run_start < result.end and run_end > result.start:
                continue
            if _WORD_CHAR.match(text, max(run_start - 1, 0), run_start) or (
                _WORD_CHAR.match(text, run_end)
            ):
                continue
            if all(
                _PHRASE_GAP_RE.fullmatch(
                    text, tokens_indices[a] + len(tokens[a]), tokens_indices[b]
                )
                for (a, _, _), (b, _, _) in zip(run, run[1:])
            ):
                logger.debug("Found context phrase '%s' in tokens", phrase)
                return True
        return False

    def _get_context_window_tokens(
        self, nlp_artifacts: NlpArtifacts, first: int, last: int
    ) -> Optional[Tuple[int, int]]:
        """Return the indices of the first and last token of the context window.

        The window reaches the tokens that `_extract_surrounding_words` and
        `_extract_words_after_entity` take their context lemmas from.

        :param nlp_artifacts: the NLP artifacts of the analyzed text
        :param first: index of the entity's first token
        :param last: index of the entity's last token
        """
        visited = [
            *self._walk_context_tokens(
                first,
                self.context_prefix_count,
                nlp_artifacts.lemmas,
                nlp_artifacts.keywords,
                is_backward=True,
            ),
            *self._walk_context_tokens(
                first,
                self.context_suffix_count,
                nlp_artifacts.lemmas,
                nlp_artifacts.keywords,
                is_backward=False,
            ),
        ]
        if self._has_words_after_entity(first, last):
            visited.extend(
                self._walk_context_tokens(
                    last + 1,
                    self.context_suffix_count - 1,
                    nlp_artifacts.lemmas,
                    nlp_artifacts.keywords,
                    is_backward=False,
                )
            )
        if not visited:
            return None
        return min(visited), max(visited)

    def _extract_surrounding_words(
        self, nlp_artifacts: NlpArtifacts, word: str, start: int
    ) -> List[str]:
        """Extract words surrounding another given word.

        The text from which the context is extracted is given in the nlp
        doc.

        :param nlp_artifacts: An abstraction layer which holds different
                              items which are the result of a NLP pipeline
                              execution on a given text
        :param word: The word to look for context around
        :param start: The start index of the word in the original text
        """
        if not nlp_artifacts.tokens:
            logger.info("Skipping context extraction due to lack of NLP artifacts")
            # if there are no nlp artifacts, this is ok, we can
            # extract context and we return a valid, yet empty
            # context
            return [""]

        # Get the already prepared words in the given text, in their
        # LEMMATIZED version
        lemmatized_keywords = nlp_artifacts.keywords

        # since the list of tokens is not necessarily aligned
        # with the actual index of the match, we look for the
        # token index which corresponds to the match
        token_index = self._find_index_of_match_token(
            word, start, nlp_artifacts.tokens, nlp_artifacts.tokens_indices
        )

        # index i belongs to the PII entity, take the preceding n words
        # and the successing m words into a context list

        backward_context = self._add_n_words_backward(
            token_index,
            self.context_prefix_count,
            nlp_artifacts.lemmas,
            lemmatized_keywords,
        )
        forward_context = self._add_n_words_forward(
            token_index,
            self.context_suffix_count,
            nlp_artifacts.lemmas,
            lemmatized_keywords,
        )

        context_list = []
        context_list.extend(backward_context)
        context_list.extend(forward_context)
        context_list = list(set(context_list))
        logger.debug("Context list is: %s", " ".join(context_list))
        return context_list

    @staticmethod
    def _find_index_of_match_token(
        word: str,
        start: int,
        tokens,
        tokens_indices: List[int],
    ) -> int:
        found = False
        # we use the known start index of the original word to find the actual
        # token at that index, we are not checking for equivalence since the
        # token might be just a substring of that word (e.g. for phone number
        # 555-124564 the first token might be just '555' or for a match like '
        # rocket' the actual token will just be 'rocket' hence the misalignment
        # of indices)
        # Note: we are iterating over the original tokens (not the lemmatized)
        i = -1
        for i, token in enumerate(tokens, 0):
            # Either we found a token with the exact location, or
            # we take a token which its characters indices covers
            # the index we are looking for.
            if (tokens_indices[i] == start) or (start < tokens_indices[i] + len(token)):
                # found the interesting token, the one that around it
                # we take n words, we save the matching lemma
                found = True
                break

        if not found:
            raise ValueError(
                "Did not find word '" + word + "' "
                "in the list of tokens although it "
                "is expected to be found"
            )
        return i

    @staticmethod
    def _find_entity_token_range(
        nlp_artifacts: NlpArtifacts, start: int, end: int
    ) -> Tuple[int, int]:
        """Return the indices of the entity's first and last token.

        The first token is the one `_find_index_of_match_token` finds: the
        first token that ends after `start`. Tokens are sorted and do not
        overlap, so a binary search over their start offsets finds it.

        :param nlp_artifacts: the NLP artifacts of the analyzed text
        :param start: the start index of the entity in the original text
        :param end: the end index of the entity in the original text
        """
        tokens = nlp_artifacts.tokens
        tokens_indices = nlp_artifacts.tokens_indices
        first = bisect_right(tokens_indices, start) - 1
        if first < 0 or tokens_indices[first] + len(tokens[first]) <= start:
            first += 1
        last = max(first, bisect_left(tokens_indices, end) - 1)
        return first, last

    def _has_words_after_entity(self, first: int, last: int) -> bool:
        """Check if the suffix window is also counted after the entity.

        `_extract_surrounding_words` counts it from the entity's first token, so
        an entity of several tokens can use it up on its own tokens.

        :param first: index of the entity's first token
        :param last: index of the entity's last token
        """
        return last > first and self.context_suffix_count > 0

    def _extract_words_after_entity(
        self, nlp_artifacts: NlpArtifacts, first: int, last: int
    ) -> List[str]:
        """Extract the `context_suffix_count` keywords after the entity's last token.

        Only consulted when the context of `_extract_surrounding_words` has no
        match, and only for an entity of several tokens.

        :param nlp_artifacts: the NLP artifacts of the analyzed text
        :param first: index of the entity's first token
        :param last: index of the entity's last token
        """
        if not self._has_words_after_entity(first, last):
            return []
        # `_add_n_words` takes one keyword more, for the entity's own token
        return self._add_n_words_forward(
            last + 1,
            self.context_suffix_count - 1,
            nlp_artifacts.lemmas,
            nlp_artifacts.keywords,
        )

    @staticmethod
    def _add_n_words(
        index: int,
        n_words: int,
        lemmas: List[str],
        lemmatized_filtered_keywords: List[str],
        is_backward: bool,
    ) -> List[str]:
        """
        Prepare a string of context words.

        Return a list of words which surrounds a lemma at a given index.
        The words will be collected only if exist in the filtered array

        :param index: index of the lemma that its surrounding words we want
        :param n_words: number of words to take
        :param lemmas: array of lemmas
        :param lemmatized_filtered_keywords: the array of filtered
               lemmas from the original sentence,
        :param is_backward: if true take the preceeding words, if false,
                            take the successing words
        """
        i = index
        context_words = []
        # The entity itself is no interest to us...however we want to
        # consider it anyway for cases were it is attached with no spaces
        # to an interesting context word, so we allow it and add 1 to
        # the number of collected words

        # collect at most n words (in lower case)
        remaining = n_words + 1
        while 0 <= i < len(lemmas) and remaining > 0:
            lower_lemma = lemmas[i].lower()
            if lower_lemma in lemmatized_filtered_keywords:
                context_words.append(lower_lemma)
                remaining -= 1
            i = i - 1 if is_backward else i + 1
        return context_words

    @staticmethod
    def _walk_context_tokens(
        index: int,
        n_words: int,
        lemmas: List[str],
        lemmatized_filtered_keywords: List[str],
        is_backward: bool,
    ) -> Iterator[int]:
        """Yield the token indices of the context window, starting at index.

        :param index: index of the lemma that its surrounding words we want
        :param n_words: number of keywords to take
        :param lemmas: array of lemmas
        :param lemmatized_filtered_keywords: the array of filtered
               lemmas from the original sentence,
        :param is_backward: if true walk to the preceeding words, if false,
                            to the successing words
        """
        i = index
        # the walk of `_add_n_words`: the entity's own token counts as one
        remaining = n_words + 1
        while 0 <= i < len(lemmas) and remaining > 0:
            yield i
            if lemmas[i].lower() in lemmatized_filtered_keywords:
                remaining -= 1
            i = i - 1 if is_backward else i + 1

    def _add_n_words_forward(
        self,
        index: int,
        n_words: int,
        lemmas: List[str],
        lemmatized_filtered_keywords: List[str],
    ) -> List[str]:
        return self._add_n_words(
            index, n_words, lemmas, lemmatized_filtered_keywords, False
        )

    def _add_n_words_backward(
        self,
        index: int,
        n_words: int,
        lemmas: List[str],
        lemmatized_filtered_keywords: List[str],
    ) -> List[str]:
        return self._add_n_words(
            index, n_words, lemmas, lemmatized_filtered_keywords, True
        )
