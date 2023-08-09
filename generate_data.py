import os
import uuid
import argparse
import random
from types import SimpleNamespace


from tqdm import tqdm
import numpy as np
from PIL import Image

import torch as th
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer
from clip_zero_shot_mosquito import ZeroShot    

def random_selector(img_dir: str, k: int = 10) -> list:
    return random.sample([os.path.join(img_dir, f) for f in os.listdir(img_dir)], k)

def detect_classes(input_folder: str):
    return [d for d in os.listdir(input_folder) 
             if os.path.isdir(os.path.join(input_folder, d))]

class FUNIT_Genrator:
    def __init__(self, cktp: str, config: str):
        self.device = 'cuda'
        config = get_config(config)
        config['batch_size'] = 1
        config['gpus'] = 1

        self.trainer = Trainer(config)
        self.trainer.to(self.device)
        self.trainer.load_ckpt(cktp)
        self.trainer.eval()

        transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [transforms.Resize((128, 128))] + transform_list
        self.transform = transforms.Compose(transform_list)

    def load_img_as_tensor(self, path: str) -> th.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img)
    
    @th.no_grad()
    def compute_style(self, style_paths: list) -> th.Tensor:
        x = []
        for path in style_paths:
            x.append(self.load_img_as_tensor(path).unsqueeze(0))
        x = th.concatenate(x, dim=0)
        return self.trainer.model.compute_k_style(x.to(self.device), len(style_paths))
    
    @th.no_grad()
    def generate_image(self, content_path: str, style_paths: list, output_path: str):

        final_class_code = self.compute_style(style_paths)

        content_img = self.load_img_as_tensor(content_path).unsqueeze(0)

        output_image = self.trainer.model.translate_simple(content_img.to(self.device), final_class_code)

        # save image
        image = output_image.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = ((image + 1) * 0.5 * 255.0)
        output_img = Image.fromarray(np.uint8(image))
        output_img.save(output_path, 'JPEG', quality=99)

def main(opts: SimpleNamespace) -> str:
    opts.output_folder = f'{opts.output_folder}_k{opts.k}_{opts.selection}_{opts.epoch:08}'
    img_gen = FUNIT_Genrator(f'./outputs/funit_mosquitos/checkpoints/gen_{opts.epoch:08}.pt', opts.config)
    zs = None
    if opts.selection == 'clip':
        clip_params = SimpleNamespace(model_name='ViT-L-14', dataset='datacomp_xl_s13b_b90k')
        zs = ZeroShot(clip_params)
        zs.index(opts.input_folder)


    classes = detect_classes(opts.input_folder)

    for src_class in classes:
        src_folder = os.path.join(opts.input_folder, src_class)
        print(f'[Generate Data] - [Main] - Start mapping {src_class}...')
        for f_name in tqdm(os.listdir(src_folder)[:opts.max_size]):
            src_path = os.path.join(src_folder, f_name)
            for dst_class in classes:
                if src_class != dst_class:
                    dst_folder = os.path.join(opts.output_folder, dst_class)
                    os.makedirs(dst_folder, exist_ok=True)
                    dst_path = os.path.join(dst_folder, f'{src_class}-{str(uuid.uuid4())}.jpeg')
                    if opts.selection == 'random':
                        style_paths = random_selector(os.path.join(opts.input_folder, dst_class), opts.k)
                    else:
                        style_paths = zs.search(src_path, dst_class, opts.k)[0]
                    img_gen.generate_image(src_path, style_paths, dst_path)
    return opts.output_folder
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/funit_mosquitos.yaml')
    parser.add_argument('--epoch',
                        type=int,
                        default=3500)
    parser.add_argument('--input_folder',
                        type=str,
                        default='../data_funit/val')
    parser.add_argument('--output_folder',
                        type=str,
                        default='../data_funit/test_generation/generated')
    parser.add_argument('--selection',
                        type=str,
                        choices=['random', 'clip'],
                        default='random')
    parser.add_argument('--k',
                        type=int,
                        default=8)
    parser.add_argument('--max_size',
                        type=int,
                        default=1000)
    opts = parser.parse_args()
    cudnn.benchmark = True
    
    opts.output_folder = f'{opts.output_folder}_k{opts.k}_{opts.selection}'
    main(opts)