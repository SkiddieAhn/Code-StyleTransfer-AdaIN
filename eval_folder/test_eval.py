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


def val_test_eval(cfg, model, iter):
    content = img2tensor(path=cfg.content_path) # (1, c, h, w)
    style = img2tensor(path=cfg.style_path) # (1, c, h, w)

    with torch.no_grad():
        content = content.cuda()
        style = style.cuda()
        output, _, _ = model(content=content, style=style)
        save_tensors_as_image([content[0], style[0], output[0]], f'{cfg.test_path}/{cfg.test_name}.jpg')
        print(f'[{cfg.test_path}/{cfg.test_name}.jpg] save ok!')
