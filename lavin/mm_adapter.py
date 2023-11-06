import torch
from torch import nn
import lavin
from typing import Optional, Tuple
from torch.cuda.amp import autocast
import lavin.eval_model


class RepAdapter_Router(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(self, in_features=768, hidden_dim=8, groups=2, scale=1, t=10., precision='fp16'):
        super().__init__()

        self.precision = {'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]

        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights = nn.Linear(in_features, 2)

        self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale
        self.t = t

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

        nn.init.zeros_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)

    def forward(self, x, weights=None):
        with autocast(dtype=self.precision):
            if weights is None:
                weights = torch.softmax(self.expert_weights(x[:, 0]) / self.t, -1)
                if self.precision == torch.float16:
                    weights = weights.half()
                elif self.precision == torch.bfloat16:
                    weights = weights.bfloat16()

            x = x.transpose(1, 2)
            x_ = self.dropout(self.conv_A(x))
            x = self.conv_B(x_) * self.scale * weights[:, 0, None, None] + self.conv_D(x_) * self.scale * weights[:, 1, None, None] + x
            x = x.transpose(1, 2).contiguous()
        return x


class RepAdapter(nn.Module):
    """
    Pytorch Implemention of RepAdapter for 1d tensor
    copy from https://github.com/luogen1996/RepAdapter/blob/main/repadapter.py
    """

    def __init__(self, in_features=768, hidden_dim=8, groups=2, scale=1, precision='fp16'):
        super().__init__()
        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.precision = {'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]

        self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x, weights=None):
        with autocast(dtype=self.precision):
            x = x.transpose(1, 2)
            x = self.conv_B(self.dropout(self.conv_A(x)))
            x = x.transpose(1, 2).contiguous()
        return x


def forward_llama_block(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        h = x + self.drop_path(
            torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
        out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.adapter_mlp(self.ffn_norm(h))))
    else:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.adapter_mlp(self.ffn_norm(h))))
    return out


def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out


def forward_llama_attn_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_ = x.shape[0]
    if start_pos == 0:
        self.cache_weights[:bs_] = torch.softmax(self.adapter_attn.expert_weights(self.attention_norm(x)[:, 0].float()) / self.t, -1).half()
    h = x + self.drop_path(
        self.attention.forward(self.adapter_attn(self.attention_norm(x), weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out


def forward_llama_attn_normal(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out


def forward_llama_attn_normal_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out


def forward_llama_block_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_ = x.shape[0]
    if start_pos == 0:
        self.cache_weights[:bs_] = torch.softmax(self.adapter_attn.expert_weights(self.attention_norm(x)[:, 0].float()) / self.t, -1).half()
        self.cache_weights_ffn[:bs_] = torch.softmax(self.adapter_mlp.expert_weights(self.ffn_norm(x)[:, 0].float()) / self.t, -1).half()
    h = x + self.drop_path(
        self.attention.forward(self.adapter_attn(self.attention_norm(x), weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.adapter_mlp(self.ffn_norm(h), self.cache_weights_ffn[:bs_])))
    return out


def forward_clip(self, x: torch.Tensor):
    x = x + self.attention(self.adapter_attn(self.ln_1(x)))
    x = x + self.mlp(self.ln_2(x))
    return x


def forward_clip_full(self, x: torch.Tensor):
    x = x + self.attention(self.adapter_attn(self.ln_1(x)))
    x = x + self.mlp(self.adapter_mlp(self.ln_2(x)))
    return x


def set_MMAdapter(model, method, dim=8, s=1, set_forward=True, t=10, gradient_checkpointing=False, precision='fp16'):
    if method == 'block':
        # not support right now
        assert NotImplementedError
        for module in model.children():
            if type(module) == lavin.model.TransformerBlock or type(module) == lavin.eval_model.TransformerBlock:
                module.adapter_attn = RepAdapter_Router(module.dim, hidden_dim=dim, scale=s, t=t, precision=precision)
                module.adapter_mlp = RepAdapter_Router(module.dim, hidden_dim=dim, scale=s, t=t, precision=precision)
                module.s = s
                module.t = t
                module.gradient_checkpointing = gradient_checkpointing
                if type(module) == lavin.eval_model.TransformerBlock:
                    bound_method = forward_llama_block_cache.__get__(module, module.__class__)
                else:
                    bound_method = forward_llama_block.__get__(module, module.__class__)
                if set_forward:
                    setattr(module, 'forward', bound_method)
            elif len(list(module.children())) != 0:
                set_MMAdapter(module, method, dim, s, set_forward=set_forward, t=t, gradient_checkpointing=False, precision=precision)

    elif method == 'attn':
        for module in model.children():
            if type(module) == lavin.model.TransformerBlock or type(module) == lavin.eval_model.TransformerBlock:
                module.adapter_attn = RepAdapter_Router(module.dim, hidden_dim=dim, scale=s, t=t, precision=precision)
                module.s = s
                module.t = t
                module.gradient_checkpointing = gradient_checkpointing
                if type(module) == lavin.eval_model.TransformerBlock:
                    bound_method = forward_llama_attn_cache.__get__(module, module.__class__)
                else:
                    bound_method = forward_llama_attn.__get__(module, module.__class__)
                if set_forward:
                    setattr(module, 'forward', bound_method)
            elif len(list(module.children())) != 0:
                set_MMAdapter(module, method, dim, s, set_forward=set_forward, t=t, gradient_checkpointing=False, precision=precision)

    elif method == 'normal':
        for module in model.children():
            if type(module) == lavin.model.TransformerBlock or type(module) == lavin.eval_model.TransformerBlock:
                module.adapter_attn = RepAdapter(module.dim, hidden_dim=dim, scale=s, precision=precision)
                module.s = s
                if type(module) == lavin.eval_model.TransformerBlock:
                    bound_method = forward_llama_attn_normal_cache.__get__(module, module.__class__)
                else:
                    bound_method = forward_llama_attn_normal.__get__(module, module.__class__)
                if set_forward:
                    setattr(module, 'forward', bound_method)
            elif len(list(module.children())) != 0:
                set_MMAdapter(module, method, dim, s, set_forward=set_forward, t=t, gradient_checkpointing=False, precision=precision)


from clip.model import ResidualAttentionBlock


def set_Clip_Adapter(model, method, dim=8, s=1, set_forward=True, t=10., precision='fp16'):
    for module in model.children():
        if type(module) == ResidualAttentionBlock:
            if method == 'router':
                module.adapter_attn = RepAdapter_Router(1024, hidden_dim=dim, scale=s, t=t, precision=precision)
            elif method == 'router_block':
                module.adapter_attn = RepAdapter_Router(1024, hidden_dim=dim, scale=s, t=t, precision=precision)
                module.adapter_mlp = RepAdapter_Router(1024, hidden_dim=dim, scale=s, t=t, precision=precision)
            elif method == 'normal':
                module.adapter_attn = RepAdapter(1024, hidden_dim=dim, scale=s, precision=precision)

            module.s = s

            if method == 'router_block':
                bound_method = forward_clip_full.__get__(module, module.__class__)
            else:
                bound_method = forward_clip.__get__(module, module.__class__)
            if set_forward:
                setattr(module, 'forward', bound_method)

        elif len(list(module.children())) != 0:
            set_Clip_Adapter(module, method, dim, s, set_forward=set_forward, t=t, precision=precision)
