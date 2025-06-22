from torch.utils.data import DataLoader, Dataset
import os
import json
import linecache
    

class MyT2IDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.prompt_list = []
        
        num = len(linecache.getlines(data_path))

        for i in range(num):
            curcontent = linecache.getline(data_path, i+1)
            self.prompt_list.append(curcontent.strip())
        
    
    def __getitem__(self, idx):
       
        return {
            'text':self.prompt_list[idx],
        }
    
    def __len__(self):
        return len(self.prompt_list)


    
class MyT2IDataset2(Dataset):
    def __init__(self, paired_data_path=None, not_paired_data_path=None, image_path=None):
        super().__init__()
        self.prompt_list = []
        self.images_list = []
        self.prompt_list2 = []
        
        allpairdata = json.load(open(paired_data_path))
        self.num1 = len(allpairdata)
        for i in range(self.num1):
            curcontent = allpairdata[i]

            self.prompt_list.append(curcontent['prompt1'].strip())
            self.prompt_list.append(curcontent['prompt2'].strip())
            self.images_list.append(os.path.join(curcontent['image1']))
            self.images_list.append(curcontent['image2'])
        
        self.num2 = len(linecache.getlines(not_paired_data_path))

        for i in range(self.num2):
            curcontent = linecache.getline(not_paired_data_path, i+1)
            self.prompt_list2.append(curcontent.strip())
        


    
    def __getitem__(self, idx):
        if idx < self.num1 // 2:
            if self.images_list[2*idx] is not None:
                return {
                    'text': [self.prompt_list[2*idx], self.prompt_list[2*idx+1]],
                    'imag': [self.images_list[2*idx], self.images_list[2*idx+1]],
                    'pair': True
                }
            else:
                return {
                    'text': [self.prompt_list[2*idx], self.prompt_list[2*idx+1]],
                    'imag': None,
                    'pair': True
                }
        else:
            idx = idx - self.num1 // 2
            return {
                    'text': [self.prompt_list2[2*idx], self.prompt_list2[2*idx+1]],
                    'imag': None,
                    'pair': False
                }

    
    
    def __len__(self):
        return self.num1 // 2 + self.num2 // 2

    
