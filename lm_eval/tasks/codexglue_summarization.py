"""
based on code by Jessy Lin
"""

import abc
import re

from lm_eval.base import rf, Task
from datasets import load_dataset
from lm_eval.metrics import weighted_mean
from lm_eval.tasks.humaneval_ppl import HumanEval

from typing import List, Tuple

from . import codexglue_summarization_evaluator

TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SPACES8 = " " * 8
EOF = "<|/ file |>"

def standardize_docstring_prompt(prefix: str, suffix: str) -> str:
    """Strips any existing docstring delimiters from the prompt prefix and suffix
    and adds our own delimiter and whitespace.

    Note lots of edge cases being handled here:
    - codexglue docstring text sometimes contains the docstring delimiters, inconsistently
    - suffix can contain other functions with docstrings
    - prefix should keep the correct indentation for the whitespace
    """
    original_delim = None

    for delim in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
        if delim in prefix:
            prefix = prefix[:prefix.index(delim)]
            original_delim = delim
            break

    # Need to be more careful about looking for single quote delimiters,
    #  since they can be used in strings
    single_single_quote_with_trailing_spaces = re.compile(r'[^\'"][\']\s*$')
    if single_single_quote_with_trailing_spaces.search(prefix):
        prefix = prefix[:single_single_quote_with_trailing_spaces.search(prefix).start()]
        original_delim = "'"

    single_double_quote_with_trailing_spaces = re.compile(r'[^\'"]["]\s*$')
    if single_double_quote_with_trailing_spaces.search(prefix):
        prefix = prefix[:single_double_quote_with_trailing_spaces.search(prefix).start()]
        original_delim = '"'

    # If we know the original delimiter, we can remove it from the suffix
    if original_delim is not None:
        if original_delim in suffix:
            suffix = suffix[suffix.index(original_delim) + len(original_delim):]
    # Delimiter not in prefix, check we don't have a delimiter in suffix
    else:
        triple_quote_with_leading_spaces = re.compile(r'^\s*(\'\'\'|""")')
        if triple_quote_with_leading_spaces.search(suffix):
            suffix = suffix[triple_quote_with_leading_spaces.search(suffix).end():]

        single_quote_with_leading_spaces = re.compile(r'^\s*[\'"]\s*\n')
        if single_quote_with_leading_spaces.search(suffix):
            suffix = suffix[single_quote_with_leading_spaces.search(suffix).end() - 1:]

    prefix += TRIPLE_QUOTE
    suffix = "\n" + suffix
    return [prefix, suffix]


def build_docstring_infill_prompt(code: str,
        docstring_text: str = None,
        standardize_docstring: bool = True,
        ) -> List[str]:
    """Splits the function into a prompt prefix and suffix for the code -> docstring infilling task.

    Args:
        code: text of the function to split
        docstring_text: exact text of the docstring if it's already in the code string and should be stripped out

    Returns:
        list of len 2, splitting code into the part before and after the docstring
    """
    assert code.startswith("def") or code.startswith("async def"), "Must be a function definition"

    if docstring_text is not None:
        # note that we will infill using whatever docstring quote used originally in the function (could be """, ''', #, ', ")
        prompt_prefix = code[:code.index(docstring_text)]
        prompt_suffix = code[code.index(docstring_text) + len(docstring_text):]
    else:
        function_def = code[:code.index(":") + 1]
        body = code[code.index(":") + 1:]
        prompt_prefix = f"{function_def}\n{SPACES4}{TRIPLE_QUOTE} "
        prompt_suffix = " {TRIPLE_QUOTE}\n{body}"

    if standardize_docstring:
        prompt_prefix, prompt_suffix = standardize_docstring_prompt(prompt_prefix, prompt_suffix)

    prompt_suffix += f"\n{EOF}"
    return [prompt_prefix, prompt_suffix]

def compute_bleu(gold_and_predicted_items: List[Tuple[str, str]]):
    """ generalization of codexglue_summarization_evaluator.computeMaps that uses lists and doesn't allow multiple references for a given instance and assumes a fixed ordering for predictinos and instances """
    predicted_map = {}
    gold_map = {}

    for ix, (gold_str, predicted_str) in enumerate(gold_and_predicted_items):
        gold, *rest = gold_str.strip().split('\t')
        if len(rest) > 0:
            print(f"warning: gold instance {ix} contains a tab; ignoring text after")
        gold_map[ix] = [codexglue_summarization_evaluator.splitPuncts(gold.strip().lower())]
  
        pred, *rest = predicted_str.strip().split('\t')
        if len(rest) > 0:
            print(f"warning: gold instance {ix} contains a tab; ignoring text after")
        predicted_map[ix] = [codexglue_summarization_evaluator.splitPuncts(pred.strip().lower())]

    return codexglue_summarization_evaluator.bleuFromMaps(gold_map, predicted_map)[0]

class CodexglueSummarization(Task, abc.ABC):
    from mosestokenizer import MosesDetokenizer
    VERSION = 0
    DATASET_PATH = "code_x_glue_ct_code_to_text"
    LANGUAGE = "python"
    DATASET_NAME = DATASET_PATH

    STOP_WORDS = ['"""', '    """']

    # "cleaned" or "full"
    # "cleaned" was used in the original paper (personal correspondence); full evals use true docstrs
    EVAL_TYPE = "cleaned"

    STANDARDIZE_DOCSTRING = True

    detokenize = MosesDetokenizer("en")

    def download(self, data_dir, cache_dir, download_mode):
        self.dataset = load_dataset(self.DATASET_PATH, self.LANGUAGE, data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)

    @classmethod
    def postprocess_model_output(cls, model_output: str):
        docstr_one_line = model_output.strip().split("\n")[0].strip()
        return docstr_one_line

    @classmethod
    def postprocess_reference(cls, dataset_record):
        # Eval with true docstrs
        if cls.EVAL_TYPE == "full":
            gold = dataset_record["docstring"].encode("unicode_escape").decode("utf-8")
            return gold
        # Eval with clean tokenized docstrings (used in paper)
        elif cls.EVAL_TYPE == "cleaned":
            #  these postprocessing steps are the ones the authors use,
            gold=' '.join(dataset_record['docstring_tokens']).replace('\n','')
            gold=' '.join(gold.strip().split())
            # apply our own detokenizer
            gold_detok = cls.detokenize(gold.split())
            return gold_detok
        else:
            raise NotImplementedError(f"invalid eval_type {cls.EVAL_TYPE}")

    def has_training_docs(self):
        """Whether the task has a training set"""
        # TODO In the future we could be more discerning. Some more recent tests have train and dev sets
        return False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        yield from self.dataset["test"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["original_string"]

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc, ctx):
        code = doc["original_string"]
        docstring_text = doc["docstring"]

        prefix, suffix = build_docstring_infill_prompt(code, docstring_text, self.STANDARDIZE_DOCSTRING)

        return rf.greedy_until(prefix, self.STOP_WORDS)

    def process_results(self, doc, results):
        assert len(results) == 1
        completion = results[0]
        gold = self.postprocess_reference(doc)
        pred = self.postprocess_model_output(completion)
        ref_pred = (gold, pred)

        # following translation.py:
        # These metrics are corpus-level not sentence level, so we'll hide the
        # results in this dict and compute the corpus score in the aggregate method
        return {
            "bleu": ref_pred,
        }

    def aggregation(self):
        return {
            "bleu": compute_bleu,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
        }
