from .gpt2 import HFLM

class InCoderLM(HFLM):
    def check_tokenizer(self, tokenizer, tokenizer_name):
        assert tokenizer_name.startswith("facebook/incoder")

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)