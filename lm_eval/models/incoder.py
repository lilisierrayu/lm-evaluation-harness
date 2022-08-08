from .gpt2 import HFLM

class InCoderLM(HFLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PAD = "<pad>"
        self.tokenizer.pad_token = PAD
        self.tokenizer.padding_side = "left"

        if kwargs.get("pretrained") == "facebook/incoder-6B":
            self.gpt2 = self.gpt2.half()

    def check_tokenizer(self, tokenizer, tokenizer_name):
        assert tokenizer_name.startswith("facebook/incoder")

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def batch_tok_encode(self, strings):
        return self.tokenizer(strings, padding=True).input_ids

    def tok_decode(self, ids):
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def batch_tok_decode(self, ids):
        return self.tokenizer.batch_decode(ids, clean_up_tokenization_spaces=False)

    @property
    def eot_token_id(self):
        # TODO: tok_encode will also prepend <|endoftext|>, so this results in
        # being forced to generate it an extra time. But doing it correctly would need
        # modifications either to base.py, or loglikelihood & loglikelihood_rolling
        return self.tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
