import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from internvl.model.internvl_chat import InternVLChatModel
import torch.nn.functional as F

import argparse


parser = argparse.ArgumentParser(description="evaluate_images")

parser.add_argument("--bsz", type=int, help="batch_size", default=2)
parser.add_argument("--image_path", default='',type=str,help="the path of the image folder")
parser.add_argument("--tgtpath", type=str,help="the path of the score json", default='prompt/score.json')

args = parser.parse_args()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'path to OpenGVLab/InternVL2_5-26B' #path to InternVL2_5-26B
model = InternVLChatModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

import functools
import os

imgpath = args.image_path

# print(ffpath)

import json
data = json.load(open('prompt/PairComp.json'))
prompt_list = []
id_list = []
id_list2 = []
id_list3 = []
image_list = []

for i in range(len(data)):
    ### add image
    assert os.path.exists(os.path.join(imgpath, f'{i}_{0}_{0}.png'))
    image_list.append(os.path.join(imgpath, f'{i}_{0}_{0}.png'))
    assert os.path.exists(os.path.join(imgpath, f'{i}_{0}_{1}.png'))
    image_list.append(os.path.join(imgpath, f'{i}_{0}_{1}.png'))
    assert os.path.exists(os.path.join(imgpath, f'{i}_{1}_{0}.png'))
    image_list.append(os.path.join(imgpath, f'{i}_{1}_{0}.png'))
    assert os.path.exists(os.path.join(imgpath, f'{i}_{1}_{1}.png'))
    image_list.append(os.path.join(imgpath, f'{i}_{1}_{1}.png'))
    ### add prompt
    prompt_list.append(data[i]['caption1'])
    prompt_list.append(data[i]['caption1'])
    prompt_list.append(data[i]['caption2'])
    prompt_list.append(data[i]['caption2'])
    for k in range(4):
        id_list.append(i)

    id_list2.append(0)
    id_list2.append(0)
    id_list2.append(1)
    id_list2.append(1)
    
    id_list3.append(0)
    id_list3.append(1)
    id_list3.append(0)
    id_list3.append(1)

bsz = args.bsz

from collections import defaultdict
final_score = defaultdict(list)

from tqdm import tqdm
for i in tqdm(range(len(prompt_list)//bsz)):

    stidx = i*bsz
    edidx = min(i*bsz+bsz, len(prompt_list))
    print(stidx, edidx)
    
    curimages = image_list[stidx:edidx]
    prompts = prompt_list[stidx:edidx]
    curidxs = id_list[stidx:edidx]
    curidxs2 = id_list2[stidx:edidx]
    curidxs3 = id_list3[stidx:edidx]
    
    # names = [os.path.join(ffpath, f'{i}_{0}_{0}.png'), os.path.join(ffpath, f'{i}_{0}_{1}.png'), os.path.join(ffpath, f'{i}_{1}_{0}.png'), os.path.join(ffpath, f'{i}_{1}_{1}.png')]
    # prompts = [prompt_list[2*i], prompt_list[2*i], prompt_list[2*i+1], prompt_list[2*i+1]]

    pixel_values = [load_image(cur_img_path, max_num=12).to(torch.bfloat16).cuda() for cur_img_path in curimages]
    num_patches_list = [pv.size(0) for pv in pixel_values]
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    pixel_values = torch.cat(pixel_values, dim=0)

    questions = ['<image>\n Does this image match the description "' + prompt + '", please directly respond with yes or no' for prompt in prompts] 
    with torch.no_grad():
      responses = model.my_batch_chat(tokenizer, pixel_values,
                                  num_patches_list=num_patches_list,
                                  questions=questions,
                                  generation_config=generation_config)

    text = tokenizer.batch_decode(responses[0], skip_special_tokens=True)
    print(text)
    logitsm = F.softmax(responses.scores[0], dim=-1).detach().cpu().squeeze(0)
    vocab_dict = tokenizer.get_vocab()
    yes_prob = (logitsm[:, vocab_dict['yes']] + logitsm[:, vocab_dict['Yes']] + logitsm[:, vocab_dict['YES']])
    no_prob = (logitsm[:, vocab_dict['no']] + logitsm[:, vocab_dict['No']] + logitsm[:, vocab_dict['NO']])
    prob_list = yes_prob / (yes_prob + no_prob)
    yes_prob_norm = [round(pp.item(), 4) for pp in prob_list]
    score = yes_prob_norm
    
    for k in range(len(prompts)):
        final_score[curidxs[k]].append({
            'pair_idx': curidxs[k],
            'prompt_idx': curidxs2[k],
            'image_idx': curidxs3[k],
            'score':score[k]
        })

# check
assert len(final_score) == len(data)
for key in final_score.keys():
    assert len(final_score[key])==4

with open(args.tgtpath, 'w', encoding='utf-8') as json_file:
    json.dump(final_score, json_file, indent=4)