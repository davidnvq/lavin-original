import torch

import json
from lavin import ModelArgs, Tokenizer, Transformer
from lavin.mm_adapter import set_MMAdapter, set_Clip_Adapter

from pathlib import Path
from lavin.utils.apply_delta import apply_model_delta_online


def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
    return checkpoint, tokenizer, params


def LaVIN(args):

    llama_model_path = args.llama_model_path
    model_name = args.llm_model

    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len,
                                      max_batch_size=32,
                                      hidden_proj=args.hidden_proj,
                                      drop_path=args.drop_path,
                                      **params)

    model_args.vocab_size = tokenizer.n_words

    if model_args.precision == 'bf16':
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    elif model_args.precision == 'fp16':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    llama = Transformer(model_args)

    #delete language encoder
    del llama.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)

    llama.load_state_dict(checkpoint, strict=False)

    set_MMAdapter(llama,
                  args.adapter_type,
                  dim=args.adapter_dim,
                  s=args.adapter_scale,
                  t=args.temperature,
                  gradient_checkpointing=False,
                  precision=model_args.precision)

    set_Clip_Adapter(llama.backbone.visual,
                     args.visual_adapter_type,
                     dim=args.adapter_dim,
                     s=args.adapter_scale,
                     t=args.temperature,
                     precision="fp16")

    learnable_keys = ['adapter']
    total = 0.
    trainable_names = []
    for name, param in llama.named_parameters():
        for key in learnable_keys:

            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total += param.nelement()
                trainable_names.append(name)
            else:
                param.requires_grad = False

    for n, p in llama.named_parameters():
        # if p.requires_grad:
        print(p.dtype, p.requires_grad, n)

    print('  + Number of trainable params: %.2fM' % (total / 1e6))
    return llama
