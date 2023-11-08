# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sys import prefix
from typing import List

import torch

from lavin.tokenizer import Tokenizer
from lavin.eval_model import Transformer
from torch.cuda.amp import autocast


class LaVIN_Generator:

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.precision = self.model.precision

        # self.backbone = clip.load('ViT-B/16', device='cpu')[0]

    def insert_image_embeds(self, text_embeds, image_embeds, prefix_img, box_embeds=None):
        # insert image, box, and indicator into the text input/label
        _bsz, seqlen, _ = text_embeds.shape
        new_examples = []

        for i, example in enumerate(text_embeds):

            # add box to inputs
            other_tokens = [prefix_img, image_embeds[i]]

            if box_embeds is not None:
                other_tokens = [box_embeds[i]] + other_tokens

            other_tokens = torch.cat(other_tokens, dim=0)  # [other-token-len, D]

            new_example = torch.cat([example[:1], other_tokens, example[1:]], dim=0)  # [BOS] [other tokens] [text tokens] [PAD] ...
            new_example = new_example[:seqlen]  # [max_len, D]

            new_examples.append(new_example)
        new_examples = torch.stack(new_examples, dim=0)  # [B, max_len, D]
        addtional_len = other_tokens.shape[0]
        return new_examples, addtional_len

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        images: torch.Tensor,
        indicators: List[int],
        max_gen_len: int,
        n_feats: int = 3,
        temperature: float = 0.8,
        top_p: float = 0.95,
        only_response: bool = False,
        batch_boxes=None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.model.eval()

        # image prefix encoding
        prefix_img = torch.tensor(self.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
        prefix_img = prefix_img.cuda()  # 3 tokens of shape [3,]
        prefix_img_embed = self.model.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)  # [3, D]

        # image encoding
        images = images.cuda()  # [B, 3, 224, 224]
        self.model.backbone.cuda()
        image_embeds = self.model.backbone.encode_image(images)
        image_embeds = self.model.adapter_proj(image_embeds)

        # box encoding
        box_embeds = None
        if self.model.has_boxes:
            box_embeds = []
            for image_boxes in batch_boxes:
                box_embed = self.model._embed_boxes(image_boxes.cuda())  # [3, D]
                box_embed = self.model._convert_dtype(box_embed)  # [3, D]
                box_embeds.append(box_embed)
            box_embeds = torch.stack(box_embeds, dim=0)  # [B, 3, D]

        # indicator encoding
        indicators = torch.Tensor(indicators).cuda().long()
        modality_embedding = self.model.adapter_modality_embedding(indicators).unsqueeze(1)

        # prompt encoding
        prompt_tokens = []
        for i, x in enumerate(prompts):
            token_idx = self.tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(token_idx)

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), 0).cuda().long()
        input_text_masks = torch.zeros_like(tokens).bool()

        # padding the token inputs of each example in the batch
        for idx, t in enumerate(prompt_tokens):
            t = t[:total_len]
            tokens[idx, :len(t)] = torch.tensor(t).long()
            input_text_masks[idx, :len(t)] = True

        token_embeds = self.model.tok_embeddings(tokens)

        # insert image, box, and indicator into the text input/labels
        token_embeds, addtional_len = self.insert_image_embeds(token_embeds, image_embeds, prefix_img_embed, box_embeds=box_embeds)

        # update the input masks
        prepend_masks = torch.ones_like(input_text_masks[:, :addtional_len], dtype=input_text_masks.dtype, device=input_text_masks.device)
        input_masks = torch.cat([prepend_masks, input_text_masks], dim=1)  # [B, addtional_len + max_len]
        input_masks = input_masks[:, :total_len]  # [B, max_len]

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):

            if prev_pos == 0:
                h = torch.cat([modality_embedding, token_embeds[:, prev_pos:cur_pos]], 1)
            else:
                h = token_embeds[:, prev_pos:cur_pos]
            logits = self.model.forward(h, prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token_embeds = torch.where(input_masks[:, cur_pos, None], token_embeds[:, cur_pos], self.model.tok_embeddings(next_token))
            token_embeds[:, cur_pos] = next_token_embeds

            next_token = torch.where(input_masks[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        decoded_responses = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

            # cut the prefix prompt
            response = t[len(prompt_tokens[i]) + addtional_len:]
            decoded_responses.append(self.tokenizer.decode(response))

        if only_response:
            return decoded, decoded_responses
        return decoded

    @torch.inference_mode()
    def generate1(
        self,
        prompts: List[str],
        images: torch.Tensor,
        indicators: List[int],
        max_gen_len: int,
        n_feats: int = 3,
        temperature: float = 0.8,
        top_p: float = 0.95,
        only_response: bool = False,
        batch_boxes=None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.model.eval()

        # image prefix encoding
        prefix_img = torch.tensor(self.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
        prefix_img = prefix_img.cuda()  # 3 tokens -> shape of [3,]
        prefix_img_embed = self.model.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)  # [3, D]

        # box encoding
        box_embeds = None
        if self.model.has_boxes:
            box_embeds = []
            for image_boxes in batch_boxes:
                box_embed = self.model._embed_boxes(image_boxes.cuda())  # [3, D]
                box_embed = self.model._convert_dtype(box_embed)  # [3, D]
                box_embeds.append(box_embed)
            box_embeds = torch.stack(box_embeds, dim=0)  # [B, 3, D]

        # text encoding
        _tokens = []
        for i, prompt in enumerate(prompts):
            token_idx = self.tokenizer.encode(prompt, bos=True, eos=False)
            _tokens.append(token_idx)

        min_prompt_size = min([len(t) for t in _tokens])
        max_prompt_size = max([len(t) for t in _tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()  # [B, max_len]
        input_text_mask = torch.zeros_like(tokens).bool()

        for idx, t in enumerate(_tokens):
            t = t[:total_len]
            tokens[idx, :len(t)] = torch.tensor(t).long()
            input_text_mask[idx, :len(t)] = True
        token_embeds = self.model.tok_embeddings(tokens)  # [B, max_len, D]

        # image encoding
        images = images.cuda()  # [B, 3, 224, 224]
        self.model.backbone.cuda()
        image_embeds = self.model.backbone.encode_image(images)  # [B, 6, d']
        image_embeds = self.model.adapter_proj(image_embeds)  # [B, 6, D]

        # box encoding
        max_boxes_per_image = 3
        box_embeds = None
        # if self.model.has_boxes:
        #     box_embeds = []
        #     for image_boxes in batch_boxes:
        #         box_embed = self.model._embed_boxes(image_boxes.cuda())  # [3, D]
        #         box_embed = self.model._convert_dtype(box_embed)  # [3, D]
        #         box_embeds.append(box_embed)
        #     box_embeds = torch.stack(box_embeds, dim=0)  # [B, 3, D]

        # [BOS] [boxes] token_embed("Image: ") [image_embed] [token_embed] [PAD] [PAD] ...
        token_embeds = self.insert_image_embeds(token_embeds, image_embeds, prefix_img_embed, box_embeds=box_embeds)

        indicators = torch.Tensor(indicators).cuda().long()  # [B,]
        modality_embedding = self.model.adapter_modality_embedding(indicators).unsqueeze(1)  # [B, 1, D]

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):

            if prev_pos == 0:
                h = torch.cat([modality_embedding, token_embeds[:, prev_pos:cur_pos]], dim=1)
            else:
                h = token_embeds[:, prev_pos:cur_pos]
            logits = self.model.forward(h, prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token_embeds = torch.where(input_text_mask[:, cur_pos, None], token_embeds[:, cur_pos], self.model.tok_embeddings(next_token))
            token_embeds[:, cur_pos] = next_token_embeds

            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        decoded_responses = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:len(_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

            # cut the prefix prompt
            response = t[len(_tokens[i]):]
            decoded_responses.append(self.tokenizer.decode(response))

        if only_response:
            return decoded, decoded_responses
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
