# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from .gpt_tokenizer import GPTTokenizer
from .duer_tokenizer import DuerTokenizer


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type), flush=True)

    assert args.tokenizer_model_path is not None

    if args.tokenizer_type == 'DuerTokenizer':
        tokenizer = _DuerTokenizer(args.tokenizer_model_path)
    else:
        tokenizer = _GPTTokenizer(args.tokenizer_model_path)

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens (new size: {})'.format(orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))
    @property
    def eos_token(self):
        raise NotImplementedError('eos_token is not provided for {} '
                                  'tokenizer'.format(self.name))
    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _GPTTokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, tokenizer_model_path, special_tokens=[]):
        name = 'GPTT '
        super().__init__(name)
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_path)
        self.tokenizer.add_tokens(special_tokens)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def eos_token(self):
        return '<|endoftext|>'

class _DuerTokenizer(AbstractTokenizer):
    def __init__(self, vocab_file, extra_special_token=[]):
        name = 'DuerT'
        super().__init__(name)

        self.tokenizer = DuerTokenizer(vocab_file, extra_special_token=extra_special_token)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        pass

    @property
    def inv_vocab(self):
        pass

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eod_id

    @property
    def eos_token(self):
        return '</s>'

