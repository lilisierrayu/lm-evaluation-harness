"""
Evaluating Large Language Models Trained on Code"
https://arxiv.org/abs/2107.03374

The HumanEval dataset of python functions and descriptions described in the
Codex paper.

This code depends on the fork of human-eval linked below, which exposes individual
function evaluation.

Homepage: https://github.com/dpfried/human-eval
"""

import abc
import re

from lm_eval.base import rf
from datasets import load_dataset
from lm_eval.metrics import weighted_mean
from lm_eval.tasks.humaneval_ppl import HumanEval

class HumanEvalGenerative(HumanEval, abc.ABC):
    STOP_WORDS = ["\nclass", "\ndef", "\n#", "\nif"]
    def construct_requests(self, doc, ctx):

        prompt = doc["prompt"]
        prompt = prompt.rstrip()
        return rf.greedy_until(prompt, self.STOP_WORDS)

    def process_results(self, doc, results):
        # results: completion string output by the model
        from human_eval.evaluation import evaluate_functional_correctness
        assert len(results) == 1
        completion = results[0]
        samples = [dict(
            task_id=doc['task_id'],
            completion=completion,
        )]
        this_scores, this_extra = evaluate_functional_correctness(sample_file=None, samples=samples, suppress=True, strict=False)
        assert len(this_scores) == 1
        pass_at_1 = this_scores['pass@1']

        return {
            "pass_at_1": (pass_at_1, 1),
        }

    def aggregation(self):
        return {
            "pass_at_1": weighted_mean,
        }

    def higher_is_better(self):
        return {
            "pass_at_1": True,
        }
