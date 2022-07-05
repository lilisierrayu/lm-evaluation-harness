import abc
import re

from lm_eval.base import rf, Task
from datasets import load_dataset
from lm_eval.metrics import weighted_mean


class HumanEvalGenerative(Task, abc.ABC):
    VERSION = 0

    DATASET_PATH = "openai_humaneval"
    DATASET_NAME = DATASET_PATH

    STOP_WORDS = ["\nclass", "\ndef", "\n#", "\nif"]

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return False

    def has_training_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def validation_docs(self):
        yield from self.dataset["test"]
                
    def train_docs(self):
        pass

    def test_docs(self):
        pass

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc, ctx):
        prompt = doc["prompt"]
        print(f"prompt: {prompt}")
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
