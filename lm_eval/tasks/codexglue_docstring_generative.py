"""

based on code by Jessy Lin
"""

import abc
import re

from typing import List

from lm_eval.base import rf, Task
from datasets import load_dataset
from lm_eval.metrics import weighted_mean
from lm_eval.tasks.humaneval_ppl import HumanEval

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

class CodexglueDocstringGenerative(Task, abc.ABC):
    VERSION = 0

    DATASET_PATH = "code_x_glue_ct_code_to_text"
    DATASET_NAME = DATASET_PATH

    STOP_WORDS = ['"""', '    """']

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
        yield from self.dataset["python"]["test"]
                
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

    def construct_requests(self, doc, ctx):
        code = doc["original_string"]
        docstring_text = doc["docstring"]

        prompt_parts = build_docstring_infill_prompt(code, docstring_text)
        prompt = prompt_parts[0]
        print(prompt)
        response = rf.greedy_until(prompt, self.STOP_WORDS)
        print(response)
        return response

    def process_results(self, doc, results):
        # results: completion string output by the model

        # TODO

        return {
            "bleu": (bleu, 1),
        }

    def aggregation(self):
        return {
            "bleu": weighted_mean,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
        }
