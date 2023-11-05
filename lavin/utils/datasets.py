# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import copy
import json, re, random
import os.path as osp
import torch
import torch.utils.data as Data

from PIL import Image
from PIL import Image, ImageDraw
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.utils.data as Data
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, RandomCrop, ColorJitter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from lavin import Tokenizer
from lavin.utils.base_prompt import *


class ScienceQADataSet(Data.Dataset):

    def __init__(self, args, split, model_path, max_words=512, max_image_feats=1):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        captions = json.load(open(args.caption_file))["captions"]
        self.image_path = os.path.join(args.data_root, 'images', split)
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats = max_image_feats
        self.split = split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def tokenize(self, prompt, answer):
        example = prompt + answer
        # print(prompt)
        prompt = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask, label_mask

    def __getitem__(self, idx):
        prompt_question, prompt_answer = build_prompt(self.problems, self.qids[idx], self.args)
        answer, choices, qid = self.problems[self.qids[idx]]["answer"], self.problems[self.qids[idx]]["choices"], self.qids[idx]

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            image_mask = torch.cat([torch.Tensor([float('-inf')] * self.max_image_feats), torch.zeros(self.max_words)])
            indicator = 1
        else:
            image = torch.Tensor(torch.zeros(3, 224, 224).float())
            image_mask = torch.zeros(self.max_words + self.max_image_feats)
            indicator = 0

        example, labels, example_mask, label_mask = self.tokenize(prompt_question, prompt_answer)

        return example, labels, example_mask, image, indicator

    def __len__(self):
        if self.args.debug:
            return 64
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)


PROMPTS = {
    'A':
        'Based on my dashcam image, what is the potential hazard? Response:',
    'B':
        'what is the potential hazard? Response:',
    'C':
        'You are a highly accurate decision-making reasoning assistant. \
        Provide reasoning hazard based on the image marked up with color boxes. \
        It is essential to distinguish between the box colors and answer questions related to them. Response:',
}


class DHPRDataset:

    def __init__(self,
                 split='train',
                 root='/home/quang/workspace/lavin-original/data',
                 max_words=128,
                 box_type='highlight',
                 n_pix=224,
                 prompt_id='A',
                 has_speed=False,
                 debug=False):
        self.debug = debug
        self.split = split
        self.root_path = root
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=osp.join(root, 'weights/tokenizer.model'))
        self.anno_data = json.load(open(osp.join(root, f'dhpr_annotations/v1101_meta_anno_{split}.json')))
        self.image_ids = list(self.anno_data.keys())

        print(f"Split {split}: {len(self.image_ids)} items!")

        self.prompt = PROMPTS[prompt_id]
        self.has_speed = has_speed

        self.box_type = box_type
        self.transforms = self._transform(n_pix)
        self.colorjitter = ColorJitter(brightness=.5, hue=.3)

    def _transform(self, n_px):
        resize_px = n_px + 16 if self.split == 'train' else n_px
        SplitCrop = RandomCrop if self.split == 'train' else CenterCrop
        return Compose([
            Resize((resize_px, resize_px), interpolation=InterpolationMode.BICUBIC),
            SplitCrop(n_px),
            ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def tokenize(self, prompt, answer):
        example = prompt + answer
        # print(prompt)
        prompt = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask, label_mask

    def hide_region(self, image, bboxes):
        COLOR_FILL = ['#ff05cd3c', '#00F1E83c', '#F2D4003c']  # 3 color - pink, green, yellow

        if self.split == 'train':
            image = self.colorjitter(image)

        image = image.convert('RGBA')
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')

        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color_fill = COLOR_FILL[idx]
            draw.rectangle([(x1, y1), (x2, y2)], fill=color_fill, outline='#05ff37ff', width=5)
        image = Image.alpha_composite(image, overlay)
        return image.convert('RGB')

    def get_item(self, image_id):
        item = self.anno_data[image_id]
        image_name = f'{image_id}.jpg' if '-' in image_id else f'{image_id}.png'

        image = Image.open(os.path.join(self.root_path, 'dhpr_images', image_name)).convert('RGB')
        image = self.hide_region(image, item['bounding_box'])

        answer = item['hazard']
        return image.convert("RGB"), answer

    def __getitem__(self, idx, do_transform=True):
        image_id = self.image_ids[idx]
        item = self.anno_data[image_id]
        image_name = f'{image_id}.jpg' if '-' in image_id else f'{image_id}.png'
        image = Image.open(os.path.join(self.root_path, 'dhpr_images', image_name)).convert('RGB')
        image = self.hide_region(image, item['bounding_box'])

        if do_transform:
            image = self.transforms(image)
        indicator = 1

        answer = item['hazard']

        prompt = self.prompt
        if self.has_speed:
            prompt = f'Our car has the plausible speed of {item["plausible_speed"]}.' + prompt
        if self.split != 'train':
            return image, indicator, prompt, answer, image_id

        example, labels, example_mask, label_mask = self.tokenize(prompt, answer)
        return example, labels, example_mask, image, indicator

    def __len__(self):
        if self.debug:
            return 64
        return len(self.anno_data)


if __name__ == '__main__':
    dataset = DHPRDataset(split='train', box_type='highlight')
    example, labels, example_mask, image, indicator = dataset[0]
    print("OK")
