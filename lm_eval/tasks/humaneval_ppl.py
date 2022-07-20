"""
Evaluating Large Language Models Trained on Code"
https://arxiv.org/abs/2107.03374

The HumanEval dataset of python functions and descriptions described in the
Codex paper.  This code was originally written by Frank Xu:
https://github.com/frankxu2004/lm-evaluation-harness/blob/ef9ef2effed9b75d118b0ebed1a62866cdc39ee3/lm_eval/tasks/humaneval_ppl.py

Homepage: https://github.com/openai/human-eval
"""
"""
"""
import abc
import re

from lm_eval.base import rf, Task, PerplexityTask
from datasets import load_dataset
from lm_eval.metrics import weighted_perplexity, token_count

def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens

class HumanEval(Task, abc.ABC):
    VERSION = 0

    DATASET_PATH = "openai_humaneval"
    DATASET_NAME = DATASET_PATH

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

    def doc_to_prompt(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return doc["canonical_solution"]

    def doc_to_text(self, doc):
        return ""

class HumanEvalPerplexity(HumanEval, PerplexityTask, abc.ABC):
    def construct_requests(self, doc, ctx):
        assert not ctx
        req = [rf.loglikelihood_rolling(self.doc_to_prompt(doc)), 
                rf.loglikelihood_rolling(self.doc_to_prompt(doc) + self.doc_to_target(doc))]
        return req

    def process_results(self, doc, results):
        prompt_loglikelihood, full_loglikelihood = results
        solution_loglikelihood = full_loglikelihood - prompt_loglikelihood
        prompt_words = self.count_words(self.doc_to_prompt(doc))
        solution_words = self.count_words(self.doc_to_target(doc))
        return {
            "prompt_bleutok_perplexity": (prompt_loglikelihood, prompt_words),
            "solution_bleutok_perplexity": (solution_loglikelihood, solution_words),
            "full_bleutok_perplexity": (full_loglikelihood, prompt_words + solution_words),
            "num_prompt_tokens": prompt_words,
            "num_solution_tokens": solution_words,
            "num_full_tokens": prompt_words + solution_words,
            # "num_prompt_model_tokens": prompt_num_model_tokens,
            # "num_solution_model_tokens": full_num_model_tokens - prompt_num_model_tokens,
            # "num_full_model_tokens": full_num_model_tokens,

        }

    def aggregation(self):
        return {
            "prompt_bleutok_perplexity": weighted_perplexity,
            "solution_bleutok_perplexity": weighted_perplexity,
            "full_bleutok_perplexity": weighted_perplexity,
            "num_prompt_tokens": token_count,
            "num_solution_tokens": token_count,
            "num_full_tokens": token_count,
            "num_prompt_model_tokens": token_count,
            "num_solution_model_tokens": token_count,
            "num_full_model_tokens": token_count,
        }

    def higher_is_better(self):
        return {
            "prompt_bleutok_perplexity": False,
            "solution_bleutok_perplexity": False,
            "full_bleutok_perplexity": False,
        }
    
    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(tokenize_for_bleu_eval(doc))
