from PIL import Image
import os
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import shutil


def img2tensor(path):
    image = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    tenosrs = image_tensor.unsqueeze(0) # (1, c, h, w)
    return tenosrs


def make_split(cfg, test_size=0.2, random_state=42):
    data_path = cfg.data_path
    parent_path = os.path.dirname(os.path.abspath(data_path))
    train_path = os.path.join(parent_path, 'train', 'class0')
    test_path = os.path.join(parent_path, 'test', 'class0')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('make_splt -> pass')

    else:
        all_files = [f for f in os.listdir(data_path)]
        train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=random_state)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for file in train_files:
            shutil.move(os.path.join(data_path, file), os.path.join(train_path, file))

        for file in test_files:
            shutil.move(os.path.join(data_path, file), os.path.join(test_path, file))

        print('make_splt -> ok')


def make_loader(cfg, batch_size=128, train=True):
    parent_path = os.path.dirname(os.path.abspath(cfg.data_path))
    if train:
        data_path = os.path.join(parent_path, 'train')
    else:
        data_path = os.path.join(parent_path, 'test')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = dset.ImageFolder(root=data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    return dataloader