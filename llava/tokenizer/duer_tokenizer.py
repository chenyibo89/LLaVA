from typing import List
import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer, AddedToken

class DuerTokenizer(PreTrainedTokenizer):
    def __init__(self,vocab_file, extra_special_token=[], **kwargs):
        super().__init__(
            bos_token = AddedToken('<duer-s>'),
            unk_token = AddedToken('<unk>'),
            eos_token = AddedToken('</duer-s>'),
            pad_token = AddedToken('<pad>'),
            cls_token = AddedToken('<cls>'),
            sep_token = AddedToken('<sep>'),
            mask_token = AddedToken('[MASK]'),
            **kwargs)

        self.additional_special_tokens = ['[gMASK]'] + extra_special_token

        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        self.sanitize_special_tokens()
        self.eod_id = self.eos_token_id

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    @property
    def gmask_token(self):
        return '[gMASK]'

    @property
    def gmask_token_id(self):
        return self.get_special_token_id(self.gmask_token)

    def get_special_token_id(self,token):
        if token in self.all_special_tokens:
            return self.all_special_ids[self.all_special_tokens.index(token)]
        else:
            raise ValueError(f'token:{token} not exists.')

    def encode_as_token(self,text,**kwargs):
        '''
            将输入的文本切分为 token 列表，而不是 id 列表
        '''
        return self._tokenize(text,**kwargs)

    def _tokenize(self, text, **kwargs):
        '''
        tokenize 调用此方法 text 切 token。
        encode函数在调用此方法时，输入文本会根据 special token 切分成片段，再分段调用。
        '''
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        '''
        convert_token_to_id 调用此方法，将单个 token 转换为 id
        '''
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        '''
        convert_id_to_token 调用此方法，将单个 id 转化为 token
        '''
        return self.sp_model.IdToPiece(index)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        '''
        格式化输入。
        args:
            token_ids_0: 指令微调的 prompt 部分
            token_ids_1： 指令微调的 answer 部分
        '''
        bos = self.bos_token_id
        eos = self.eos_token_id
        gmask = self.gmask_token_id

        final_token_ids = token_ids_0 + [eos]
        if token_ids_1:
            final_token_ids += [gmask] + token_ids_1 + [eos]

        return final_token_ids

    def convert_tokens_to_string(self, tokens):
        '''
        将 piece 合成为 text。
        pieces中含有"\u2581"、bytes等特殊字符。此方法将特殊字符decode为原始字符串
        '''
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):

            if token in self.all_special_tokens:
                if False and not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False

        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string
