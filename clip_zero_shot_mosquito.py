import os
from types import SimpleNamespace
from typing import Tuple
import json

import open_clip
from open_clip import CLIP
from torchvision.transforms import Compose, ToPILImage

import faiss
import torch as th
import numpy as np
import cv2
from tqdm import tqdm

def load_clip_model(model_name:str, dataset:str) -> Tuple[CLIP, Compose]:
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, 
                                                                        pretrained=dataset,
                                                                        precision='fp16')
    clip_model.eval()
    preprocess = Compose([ToPILImage(), preprocess])
    return clip_model, preprocess


class ZeroShot:
    def __init__(self, clip_params: SimpleNamespace):
        self.clip_params = clip_params
        self.device = 'cuda'

        self.faiss_index = None
        self.data = None
        self.clip_model = None
        self.preprocess = None
        self.cache = 'cache'
        os.makedirs(self.cache, exist_ok=True)

    def load_image(self, img_path: str) -> th.Tensor:
        # load img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocess
        x = self.preprocess(img)
        x = th.unsqueeze(x, 0) 

        return x
     
    @th.no_grad()
    def index(self, folder: str):



        self.clip_model, self.preprocess = load_clip_model(self.clip_params.model_name, self.clip_params.dataset)
        self.clip_model.to(self.device)

        self.faiss_index = {}
        self.data = {}

        for label in os.listdir(folder):
            label_folder = os.path.join(folder, label)
            print(f'[ZeroShot] - [Index] - Label {label_folder}...')

            emb_cache = os.path.join(self.cache, f'{self.clip_params.model_name}-{self.clip_params.dataset}-{label}.npy')

            embeddings = [] 

            for f in tqdm(os.listdir(label_folder)):
                f_path = os.path.join(label_folder, f)

                if not os.path.exists(emb_cache):  
                    x = self.load_image(f_path)
                    emb = self.clip_model.encode_image(x.half().to(self.device)).data.cpu().numpy()
                    embeddings.append(emb)

                tmp = self.data.get(label, [])
                self.data[label] = tmp + [f_path]

            if not os.path.exists(emb_cache):
                embeddings = np.concatenate(embeddings, axis=0)
                np.save(emb_cache, embeddings)

            if  os.path.exists(emb_cache):
                embeddings = np.load(emb_cache)


            self.faiss_index[label] = faiss.IndexFlatL2(embeddings.shape[1])
            self.faiss_index[label].add(embeddings)
            print(f'[ZeroShot] - [Index] - FAIS INDEX {label} is added...')


    @th.no_grad()
    def search(self, img_path: str, faiss_index_name: str, k: int = 5) -> tuple:
        label = []
        dst = []

        if self.faiss_index is not None:
            if self.clip_model is None or self.preprocess is None:
                self.clip_model, self.preprocess = load_clip_model(self.clip_params.model_name, self.clip_params.dataset)
                self.clip_model.to(self.device)

            # load img
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocess
            x = self.preprocess(img)
            x = th.unsqueeze(x, 0)

            # get emb
            emb = self.clip_model.encode_image(x.half().to(self.device))
            emb = emb.detach().cpu().numpy()

            dst, ind = self.faiss_index[faiss_index_name].search(emb, k)
            dst = dst[0]
            label = [self.data[faiss_index_name][i] for i in ind[0]]

        return label, dst

if __name__ == '__main__':
    clip_params = SimpleNamespace(model_name='ViT-L-14', dataset='datacomp_xl_s13b_b90k')
    folder = '../data_funit/test_generation/test'



    zs = ZeroShot(clip_params)
    zs.index(folder)

    img_path = '../data_funit/test_generation/test/albopictus/0a935b67-a637-4e68-8d9c-51fa7881f818.jpeg'
    label = 'aegypti'
    label, dst =  zs.search(img_path, label)
    print(dst)
    print(label)
    print('--------------------')
    