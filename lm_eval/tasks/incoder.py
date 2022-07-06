import os
import re
import json
import abc
from pathlib import Path
import gzip
import glob

from pygments import lexers

from lm_eval.base import rf, PerplexityTask
from lm_eval.utils import sh

from transformers import AutoTokenizer


class Incoder(PerplexityTask, abc.ABC):
    VERSION = 0
    LANG_NAME = None
    FILEGLOB = None

    MAX_TOKENS = None
    TOKENIZER = None

    def download(self):
        self.lexer = lexers.get_lexer_by_name(self.LANG_NAME.lower())

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return False

    def has_test_docs(self):
        return False

    def process_file(self, filename):
        buffer = None
        inside_file = False
        if filename.endswith(".gz"):
            f = gzip.open(filename, 'rt')
        else:
            f = open(filename, 'r')
        for line in f:
            if line.startswith("<| file"):
                inside_file = True
                buffer = ""
            elif line.startswith("<|/ file"):
                inside_file = False
                yield buffer
                buffer = ""
            elif inside_file:
                buffer += line
        f.close()
        if buffer:
            print(f"warning: non-empty buffer (length {len(buffer)}) at end of {filename}")

    
    def validation_docs(self):
        for file in glob.glob(self.FILEGLOB):
            yield from self.process_file(file)
                
    def train_docs(self):
        pass

    def test_docs(self):
        pass

    def doc_to_target(self, doc):
        if self.MAX_TOKENS is not None:
            tokens = self.TOKENIZER.encode(doc)[:self.MAX_TOKENS]
            doc = self.TOKENIZER.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return doc
    
    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(list(self.lexer.get_tokens(doc)))

class IncoderPython(Incoder):
    LANG_NAME = "Python"
    FILEGLOB = "/checkpoint/dpf/data/processed_py_filenames_redact/raw/val_*"

class IncoderPythonShort(Incoder):
    LANG_NAME = "Python"
    # NOTE: this only contains python files
    FILEGLOB = "/checkpoint/dpf/data/code_corpus/processed_py_short_filenames_redact/raw/val_*"

class IncoderPythonGH2(Incoder):
    LANG_NAME = "Python"
    # NOTE: this also contains non-python files
    FILEGLOB = "/checkpoint/dpf/data/processed_filenames_redact_2/raw/val_github_python_forkless_open-source_2star+_0.*"

class IncoderPythonGH2_2048(Incoder):
    LANG_NAME = "Python"
    # NOTE: this also contains non-python files
    FILEGLOB = "/checkpoint/dpf/data/processed_filenames_redact_2/raw/val_github_python_forkless_open-source_2star+_0.*"

    MAX_TOKENS = 2048
    TOKENIZER = AutoTokenizer.from_pretrained("facebook/incoder-1B")