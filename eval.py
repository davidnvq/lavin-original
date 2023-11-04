# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from datetime import datetime

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter, set_Clip_Adapter
from util.base_prompt import build_prompt
from dataclasses import dataclass
import re
import random

import warnings
import pandas as pd
from PIL import Image

from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from util.apply_delta import apply_model_delta_online

warnings.filterwarnings('ignore')


@dataclass
class PromptArgs:
    prompt_format = 'QCM-ALE'
    use_caption = True
    options = ["A", "B", "C", "D", "E"]


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


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    results = json.load(open(result_file))
    num = len(results)
    print(f"Number of predictions: {num}!")
    # assert num == 4241

    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():

        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100

    scores = {
        'acc_natural': get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social': get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language': get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text': get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image': get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context': get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6': get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12': get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average': "{:.2f}".format(acc_average),
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


def load(
    ckpt_dir: str,
    llm_model: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    adapter_type: str,
    adapter_dim: int,
    adapter_scale: float,
    hidden_proj: int,
    visual_adapter_type: str,
    temperature: float,
    use_vicuna: bool,
    bits: str = '16bits',
    cpu_load: bool = False,
) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, hidden_proj=hidden_proj, **params)
    model_args.vocab_size = tokenizer.n_words

    # if cpu_load:
    #     #cpu load is slow, but is freindly for GPU with limited memory.
    #     torch.set_default_tensor_type(torch.HalfTensor)
    # else:
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    #delete language encoder
    del model.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)

    if bits in ['4bit', '8bit']:
        from util.quantization import quant_model_bnb
        model.layers = quant_model_bnb(model.layers, quant_bit='4bit')

    set_MMAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale, t=temperature)
    set_Clip_Adapter(model.backbone.visual, visual_adapter_type, dim=adapter_dim, s=adapter_scale, t=temperature)

    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model, '../data/weights/vicuna_' + llm_model)

    state_dict = {}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.', '')] = adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    for name, param in model.named_parameters():
        print(name, param.dtype)
    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    data_root: str,
    caption_file: str,
    max_seq_len: int,
    max_batch_size: int,
    llm_model: str = '7B',
    generation_temperature: float = 0.1,
    top_p: float = 0.75,
    split='val',
    prompt_format='QCM-ALE',
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    adapter_type='repattn',
    adapter_dim=8,
    adapter_scale=1,
    n_prompt=10,
    hidden_proj=128,
    visual_adapter_type='normal',
    temperature=10.,
    use_vicuna=False,
    bits: str = '16bits',
    cpu_load: bool = False,
    output_dir: str = None,
    debug: bool = False,
    wandb_name: str = '',
):
    print(max_batch_size, max_seq_len)
    print('use caption: ', use_caption)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    if wandb_name != '':
        import wandb
        wandb.init(project="lavin-original", name=wandb_name)

    generator = load(ckpt_dir,
                     llm_model,
                     tokenizer_path,
                     adapter_path,
                     local_rank,
                     world_size,
                     max_seq_len,
                     max_batch_size,
                     adapter_type,
                     adapter_dim,
                     adapter_scale,
                     hidden_proj,
                     visual_adapter_type,
                     temperature,
                     use_vicuna,
                     bits=bits,
                     cpu_load=cpu_load)

    print('split: ', split)
    problems = json.load(open(os.path.join(data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(data_root, 'pid_splits.json')))
    captions = json.load(open(caption_file))["captions"]
    image_path = os.path.join(data_root, 'images', split)
    qids = pid_splits['%s' % (split)]
    total_items = len(qids)
    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""
    print('total_items: ', total_items)

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    prompt_args = PromptArgs()
    prompt_args.prompt_format = prompt_format
    prompt_args.use_caption = use_caption
    prompt_args.options = options

    pattern = re.compile(r'The answer is ([A-Z]).')

    answers = []
    preds = []

    if debug:
        total_items = total_items // 50
    print("Total items: ", total_items, "Debug: ", debug)

    for i in range(total_items // max_batch_size + 1):
        print('progresses: ', i, ' / ', total_items // max_batch_size + 1)
        batch_qids = qids[i * max_batch_size:(i + 1) * max_batch_size]
        if len(batch_qids) == 0:
            break
        indicators = []
        prompts = []
        images = []
        for qid in batch_qids:
            prompt, _ = build_prompt(problems, qid, prompt_args)

            answer = problems[qid]["answer"]
            if problems[qid]['image'] is not None:
                image = Image.open(os.path.join(image_path, qid, 'image.png')).convert('RGB')
                image = image_transforms(image)
                indicator = 1
            else:
                image = torch.Tensor(torch.zeros(3, 224, 224).float())
                indicator = 0
            prompts.append(prompt)
            answers.append(answer)
            images.append(image.unsqueeze(0))
            indicators.append(indicator)
        images = torch.cat(images, 0)

        results = generator.generate(prompts,
                                     images=images,
                                     indicators=indicators,
                                     max_gen_len=64,
                                     temperature=generation_temperature,
                                     top_p=top_p,
                                     n_feats=n_prompt)

        for result in results:
            pred = pattern.findall(result)

            if len(pred) >= 1:
                pred = pred[0]  # 'A', 'B', ...
            else:
                print(result)
                pred = "FAILED"
            preds.append(pred)

    #evaluations
    results = {}
    correct = 0
    for i, prediction in enumerate(preds):
        pred_idx = get_pred_idx(prediction, problems[qids[i]]["choices"], prompt_args.options)  # 0, 1, ..., 4
        if pred_idx == answers[i]:
            correct += 1
        results[qids[i]] = pred_idx
    acc = correct / len(results) * 100
    print('overall accuracy: ', acc)

    current_time = str(datetime.now().strftime("%M_%S"))

    filename = f'debug_preds_{current_time}.json' if debug else f'preds_{current_time}.json'
    if output_dir is None:
        output_file = os.path.join('outputs', filename)
    else:
        output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

    scores = get_scores(output_file, os.path.join(data_root, 'problems.json'))
    with open(output_file, 'a') as f:
        f.write(str(scores))
        print(scores)


if __name__ == "__main__":
    fire.Fire(main)
