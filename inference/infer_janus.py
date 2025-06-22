import numpy as np
import json
import torch
import sys
sys.path.append('.')
from janus.models.processing_vlm import VLChatProcessor
from janus.models.myjanus import JanusLLamaModel
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  
import os


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="t2i")

    parser.add_argument("--cfg", type=float, default=5.0)

    parser.add_argument("--processor_path", type=str, default="/mnt/prev_nas/refine_draw/models/deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--model_path", type=str, default="/mnt/prev_nas/refine_draw_RL_t2i/output/test/Janus-t2i-v0509-1.0/checkpoint-100")
    parser.add_argument("--num_gen", type=int, default=2)
    parser.add_argument("--prompt_path", type=str, default='/mnt/prev_nas/refine_draw/code/RefineDraw/data_construct/prompt/PairComp.json')
    parser.add_argument("--save_path", type=str, default='/ossfs/workspace/imgs')
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    data = json.load(open(args.prompt_path))
    prompt_list = []
    id_list = []
    id_list2 = []
    for i in range(len(data)):
        prompt_list.append(data[i]['caption1'])
        prompt_list.append(data[i]['caption2'])
        id_list.append(i)
        id_list.append(i)
        id_list2.append(0)
        id_list2.append(1)
    
    input_ids  = []
    use_pro_prompt_format = False
    
    model_path = args.model_path
    os.makedirs(args.save_path, exist_ok=True)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.processor_path)
    tokenizer = vl_chat_processor.tokenizer
    mymodel = JanusLLamaModel.from_pretrained(model_path, trust_remote_code=True)
    print('set model!')

    mymodel = mymodel.to(torch.bfloat16).cuda().eval()

    import math
    batch_size = args.batch_size
    upper = math.ceil(len(prompt_list)/batch_size) 
    
    from tqdm import tqdm
    for i in tqdm(range(0, upper)):
        st, ed = i*batch_size, min(i*batch_size+batch_size, len(prompt_list))
        
        curprompts = []
        for k in range(st, ed):
            
            conversation = [
                {
                    "role": "<|User|>",
                    "content":prompt_list[k],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + vl_chat_processor.image_start_tag
            curprompts.append(prompt)
        
        tokenizer.padding_side = 'left'
        instruction = tokenizer(
            curprompts,
            return_tensors="pt",
            padding='longest',
        ).to('cuda')
        print(curprompts[:2])
        bsz, L, dtype = instruction['input_ids'].size(0), instruction['input_ids'].size(1), instruction['input_ids'].dtype
        
        prompt_ids = instruction['input_ids']
        prompt_mask = instruction['attention_mask']
        
        if args.num_gen > 1:
            prompt_mask = torch.repeat_interleave(prompt_mask, args.num_gen, dim=0) 
            prompt_ids = torch.repeat_interleave(prompt_ids, args.num_gen, dim=0) 

        with torch.no_grad():         
            _, images, _ = mymodel.generate_image(vl_chat_processor=vl_chat_processor,
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                cur_step=i,
                set_cfg=True,cfg_weight=args.cfg)
        
        for k, (p1,p2) in enumerate(zip(id_list[st:ed], id_list2[st:ed])):
            for j, image in enumerate(images[k * args.num_gen : (k + 1) * args.num_gen]):
                image_save_path = os.path.join(args.save_path, f'{p1}_{p2}_{j}.png')
                image.save(image_save_path)

        
