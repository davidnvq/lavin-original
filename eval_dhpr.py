# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import torch
import fire
import time
import json
import datetime

import wandb
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter, set_Clip_Adapter
from lavin.utils.base_prompt import build_prompt
from dataclasses import dataclass, asdict
import re

from aac_metrics import evaluate

import warnings
from lavin.utils.datasets import DHPRDataset, dhpr_collate

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from train_dhpr import TrainArgs

warnings.filterwarnings('ignore')


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    # initialize_model_parallel(world_size)
    # single GPU only!
    initialize_model_parallel(model_parallel_size_=1)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
    print('Finished loading model path: %s, model_name: %s to CPU' % (llama_model_path, model_name))
    return checkpoint, tokenizer, params


def load(checkpoint, tokenizer, model_params, adapter_checkpoint, args):
    start_time = time.time()

    model_args = ModelArgs(
        max_seq_len=256,
        max_batch_size=4,
        hidden_proj=args.hidden_proj,
        has_boxes=args.has_boxes,
        **model_params,
    )
    model_args.vocab_size = tokenizer.n_words

    if model_args.precision == 'bf16':
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    elif model_args.precision == 'fp16':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_args)
    #delete language encoder
    del model.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)

    set_MMAdapter(
        model,
        args.adapter_type,
        dim=args.adapter_dim,
        s=args.adapter_scale,
        t=args.temperature,
        precision=model_args.precision,
        num_routers=args.num_routers,
        weight_kind=args.weight_kind,
    )
    set_Clip_Adapter(
        model.backbone.visual,
        args.visual_adapter_type,
        dim=args.adapter_dim,
        s=args.adapter_scale,
        t=args.temperature,
        precision="fp16",
    )

    model.load_state_dict(checkpoint, strict=False)

    state_dict = {}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.', '')] = adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def redefine_print():
    import builtins
    builtin_print = builtins.print

    def print(*args, **kwargs):
        now = datetime.datetime.now().time()
        builtin_print('Eval [{}] '.format(now), end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print


@dataclass
class EvalArgs(TrainArgs):
    adapter_path = "./outputs/exp1_dhpr_7b01_gt4/checkpoint-19.pth"
    generation_temperature: float = 0.0
    top_p: float = 0.75


def get_name(filename="checkpoint-12.pth"):
    # return ckpt12
    # Use regular expression to extract numeric part
    match = re.search(r'\d+', filename)
    if match:
        numeric_part = match.group()
        new_filename = "ckpt" + numeric_part
        return new_filename
    else:
        print("No numeric part found in the filename.")
        return filename


def main(adapter_path="./outputs/exp1_dhpr_7b01_gt4/checkpoint-19.pth", has_boxes=False, **kwargs):
    local_rank, world_size = setup_model_parallel()
    redefine_print()

    # load adapter & train_args
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    train_args = adapter_checkpoint['args']
    train_args.update(kwargs)

    eval_args = EvalArgs(**train_args)
    eval_args.adapter_path = adapter_path

    for k, v in asdict(eval_args).items():
        print(f"{k:<20}: {v}")

    proj_name = os.path.basename(os.path.dirname(adapter_path))
    ckpt_name = os.path.basename(adapter_path).split('.')[0]
    new_ckpt_name = get_name(ckpt_name)

    if (not eval_args.debug) and eval_args.wandb_enable:
        wandb.init(project="lavin-original",
                   name='e7-' + new_ckpt_name + '-' + proj_name,
                   dir=os.path.dirname(adapter_path),
                   config=asdict(eval_args))

    checkpoint, tokenizer, model_params = _load_and_redistribute_checkpoint(eval_args.llama_model_path, eval_args.llm_model)
    generator = load(checkpoint, tokenizer, model_params, adapter_checkpoint, eval_args)

    for split in ['val', 'test']:
        print('split: ', split)

        dataset = DHPRDataset(split=split, max_words=256, has_boxes=eval_args.has_boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=eval_args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=dhpr_collate,
        )

        ret = {}
        predictions = []
        mult_references = []

        out_path = os.path.join(os.path.dirname(eval_args.adapter_path), f'{ckpt_name}-{split}-predictions.json')
        score_path = os.path.join(os.path.dirname(eval_args.adapter_path), f'{ckpt_name}-{split}-score.json')

        def save_outputs(predictions, mult_references):
            metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "cider_d"]
            corpus_scores, _ = evaluate(predictions, mult_references, metrics=metrics)
            corpus_scores = {k: v.item() for k, v in corpus_scores.items()}
            print(idx, str(corpus_scores), '\n-----\n')
            with open(score_path, 'w') as f:
                json.dump(corpus_scores, f)

            with open(out_path, 'w') as f:
                json.dump(ret, f)

            corpus_scores = {f"{split[:]}_{k}": v for k, v in corpus_scores.items()}
            if not eval_args.debug:
                wandb.log(corpus_scores)

        total_batches = len(dataloader)

        if eval_args.debug:
            total_batches = 4  # len(dataloader)

        start_time = time.time()
        for idx, (images, indicators, prompts, gt_answers, image_ids, batch_boxes) in zip(range(total_batches), dataloader):
            preds, responses = generator.generate(
                prompts,
                images=images,
                indicators=indicators,
                max_gen_len=128,
                temperature=eval_args.generation_temperature,
                top_p=eval_args.top_p,
                n_feats=eval_args.n_prompt,
                only_response=True,
                batch_boxes=batch_boxes,
            )

            for pred, response, image_id, gt_answer in zip(preds, responses, image_ids, gt_answers):
                ret[image_id] = {'pred': pred, 'response': response, 'gt_answer': gt_answer}
                print('\n')
                print(f'image_id: ', image_id)
                print('response: ', response)
                print('gt_answer: ', gt_answer)

            predictions = predictions + [response.strip().replace('\n', '') for response in responses]
            mult_references = mult_references + [[gt.strip().replace('\n', '')] for gt in gt_answers]

            iter_time = time.time() - start_time
            eta_time = iter_time * (total_batches - idx)
            eta_time = str(datetime.timedelta(seconds=eta_time))
            print("\n\n")
            print(f"Batch idx [{idx}/{total_batches}] Eta: {eta_time} \n\n")
            start_time = time.time()

        save_outputs(predictions, mult_references)

        if eval_args.debug:
            break


if __name__ == "__main__":
    fire.Fire(main)
