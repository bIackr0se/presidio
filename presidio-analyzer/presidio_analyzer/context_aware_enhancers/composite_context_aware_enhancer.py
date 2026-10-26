import copy
import logging
from typing import List, Optional, Set

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.context_aware_enhancers.context_aware_enhancer import (
    ContextAwareEnhancer,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

logger = logging.getLogger("presidio-analyzer")


def _defining_class(cls: type, name: str) -> type:
    return next(klass for klass in cls.__mro__ if name in vars(klass))


class CompositeContextAwareEnhancer(ContextAwareEnhancer):
    """
    Run several context aware enhancers in order; the first match wins.

    Once an enhancer finds a supportive context word for a result, the later
    enhancers skip that result, so its score is enhanced at most once.
    The results are copied once for all the enhancers.

    :param enhancers: the enhancers to run, in order
    """

    def __init__(self, enhancers: List[ContextAwareEnhancer]):
        self.enhancers = list(enhancers)

    def enhance_using_context(
        self,
        text: str,
        raw_results: List[RecognizerResult],
        nlp_artifacts: NlpArtifacts,
        recognizers: List[EntityRecognizer],
        context: Optional[List[str]] = None,
    ) -> List[RecognizerResult]:
        """
        Update results with the first of the enhancers that finds context.

        :param text: The actual text that was analyzed
        :param raw_results: Recognizer results which didn't take
                            context into consideration
        :param nlp_artifacts: The nlp artifacts of the analyzed text
        :param recognizers: the list of recognizers
        :param context: list of context words
        """
        results = copy.deepcopy(raw_results)
        self._enhance_in_place(text, results, nlp_artifacts, recognizers, context)
        return results

    def _enhance_in_place(
        self,
        text: str,
        results: List[RecognizerResult],
        nlp_artifacts: NlpArtifacts,
        recognizers: List[EntityRecognizer],
        context: Optional[List[str]] = None,
    ) -> Set[int]:
        """Enhance results in place; return the positions whose context matched."""
        pending = list(range(len(results)))
        for enhancer in self.enhancers:
            if not pending:
                break
            subset = [results[i] for i in pending]
            args = (text, subset, nlp_artifacts, recognizers, context)
            if self._overrides_enhance_using_context(enhancer):
                # the base step runs its enhance_using_context, so the override
                # is not bypassed
                matched = ContextAwareEnhancer._enhance_in_place(enhancer, *args)
            else:
                matched = enhancer._enhance_in_place(*args)
            for i, result in zip(pending, subset):
                results[i] = result
            pending = [i for k, i in enumerate(pending) if k not in matched]
        return set(range(len(results))) - set(pending)

    @staticmethod
    def _overrides_enhance_using_context(enhancer: ContextAwareEnhancer) -> bool:
        """Check if enhance_using_context and _enhance_in_place differ in class.

        This is the case for a custom enhancer, and for a subclass of a built-in
        enhancer that overrides only enhance_using_context.
        """
        public = _defining_class(type(enhancer), "enhance_using_context")
        in_place = _defining_class(type(enhancer), "_enhance_in_place")
        return public is not in_place
