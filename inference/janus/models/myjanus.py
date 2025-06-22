import torch
import numpy as np

from .modeling_vlm import MultiModalityCausalLM
from .processing_vlm  import VLChatProcessor

from PIL import Image



class JanusLLamaModel(MultiModalityCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config
  
    @torch.no_grad()
    def generate_image(
        self,
        vl_chat_processor: VLChatProcessor,
        input_ids,
        attention_mask,
        set_cfg=True,
        temperature: float = 1,
        parallel_size: int = 16,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        parallel_size = input_ids.shape[0]
        if set_cfg:
            tokens = torch.repeat_interleave(input_ids,2,dim=0)
            for i in range(tokens.size(0)):
                if i % 2 != 0:
                    pad_list = torch.where(tokens[i]==vl_chat_processor.pad_id)[0]
                    if pad_list.shape[0]==0:
                        st = 1
                    else:
                        st = pad_list[-1].item()+2
                    tokens[i, st:-1] = vl_chat_processor.pad_id
                    
            attention_mask = torch.repeat_interleave(attention_mask, 2, dim=0) 
        else:
            tokens = input_ids

        inputs_embeds = self.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
        B = attention_mask.shape[0]
        from tqdm import tqdm
        for i in tqdm(range(image_token_num_per_image)):
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.gen_head(hidden_states[:, -1, :])
            if set_cfg:
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
            
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            # import pdb
            # pdb.set_trace()
            if set_cfg:
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(B, 1).to(attention_mask)], dim=1)

        dec = self.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        final_imgs = [Image.fromarray(img) for img in visual_img]

        return generated_tokens, final_imgs, (tokens, attention_mask)
    


