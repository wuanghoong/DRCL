import torch
import os.path as osp
import json


def get_raw_text_wikics(use_text=False, seed=0):
    if osp.exists(f"./datasets/wikics/wikics.pt"):
        data = torch.load(f"./datasets/wikics/wikics.pt", map_location='cpu')
        data.train_mask = data.train_mask[:,seed]
        data.val_mask = data.val_mask[:,seed]
        # data.test_mask = data.test_masks[seed]
        # with open('./datasets/wikics.json', 'w', encoding='utf-8') as f:
        #     json.dump(data.raw_texts, f, ensure_ascii=False, indent=2)
        return data, data.raw_texts
    else:
        raise NotImplementedError('No existing wikics dataset!')