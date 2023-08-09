import os

import torch as th
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import torch.backends.cudnn as cudnn

from pytorch_fid.fid_score import calculate_frechet_distance, ImagePathDataset
from pytorch_fid.inception import InceptionV3

cudnn.benchmark = True


device = 'cuda'
dims = 768
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx], resize_input=False).to(device)
model.eval()

def get_files(folder: str) -> list:
    res = []
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[-1] in ['.jpeg', '.JPEG', '.jpg', '.png']:
                res.append(os.path.join(root, name))

    return res

@th.no_grad()
def _calculate_activation_statistics(folder, batch_size=256, num_workers=8):
    global model, device, dims
    
    files = get_files(folder)
    dataset = ImagePathDataset(files,
                               transforms=T.Compose([
                                   T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                                   T.ToTensor()
                                ])
    )
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)

    act = np.empty((len(files), dims))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        pred = model(batch)[0]
        
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        act[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def compute_fid(folder1: str, folder2: str) -> float:
    global model, device, dims
    
    files1 = get_files(folder1)
    print(f'total files in {folder1} - {len(files1)}')
    m1, s1 = _calculate_activation_statistics(files1, 256)

    files2 = get_files(folder2)
    print(f'total files in {folder2} - {len(files2)}')
    m2, s2 = _calculate_activation_statistics(files2, 256)

    return calculate_frechet_distance(m1, s1, m2, s2)


def plot_fids_scores(epochs: list, fids: dict, best_fid: float, title: str):
    plt.clf()
    plt.cla()

    plt.title(title)
    for key, value in fids.items():
        plt.plot(epochs, value, '-o', label=f'CLIP({key})-FID')

    plt.plot([epochs[0]-10, epochs[-1]+10], [best_fid, best_fid], '--k', label='Best FID')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title}.png')

if __name__ == '__main__':
    
    from generate_data import main
    import matplotlib.pyplot as plt
    from types import SimpleNamespace



    real_dataset2 = '../data_funit/val'
    real_dataset1 = '../data_funit/train'
    
    # m_train, s_train = _calculate_activation_statistics(real_dataset1)
    m_val, s_val = _calculate_activation_statistics(real_dataset2)

    # best_fid = calculate_frechet_distance(m_train, s_train, m_val, s_val)
    best_fid = 0
    
    epochs = [20000, 25000, 30000, 35000, 40000, 45000]
    ks = [1, 5, 10, 15, 30]
    
    fids = {}
    for k in ks:
        _fids = []
        for epoch in epochs:
            opts = SimpleNamespace(
                epoch=epoch,
                selection='clip',
                k=k,
                input_folder=real_dataset1,
                output_folder='../data_funit/test_generation/generated',
                config='configs/funit_mosquitos.yaml',
                max_size=100
            )
            generated_dataset = main(opts)

            m_fake, s_fake = _calculate_activation_statistics(generated_dataset)
            _fids.append(calculate_frechet_distance(m_fake, s_fake, m_val, s_val))
            print(f'{epoch} | {k} | {_fids[-1]}')
        fids[f'k={k}'] = _fids
    plot_fids_scores(epochs, fids, best_fid, 'CLIP Selection-FUNIT')
    # fid_real = compute_fid(real_dataset1, generated_dataset)
    # print(fid_real)

# 3.2  - val, fake (clip)
# 2.6  - train, fake (clip)
# 0.7  - train, val
# 3.1  - val, fake (random)
# 2.6  - train, fake (random)