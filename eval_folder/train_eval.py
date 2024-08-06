from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import dataset
import warnings
from dataset import *
warnings.filterwarnings('ignore')


def save_tensors_as_image(tensors, file_name):
    images = [np.transpose(tensor.detach().cpu().numpy(), (1, 2, 0)) for tensor in tensors]
    titles = ["content", "style", "output"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(file_name, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


def val_train_eval(cfg, model, iter):
    # dataloader
    test_loader = make_loader(cfg, batch_size=1, train=False)
    style = img2tensor(path=cfg.style_path) # (1, c, h, w)

    r = random.randint(1, 100)

    with torch.no_grad():
        for i, content in enumerate(test_loader):
            if i == r:
                content = content[0].cuda()
                style = style.cuda()
                output, _, _ = model(content=content, style=style)
                save_tensors_as_image([content[0], style[0], output[0]], f'{cfg.results_path}/{iter}iter_img.jpg')
                print(f'[{cfg.results_path}/{iter}iter_img.jpg] save ok!')
                break