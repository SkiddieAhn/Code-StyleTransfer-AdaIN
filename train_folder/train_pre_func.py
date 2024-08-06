import torch 
import argparse
import numpy as np
from model.net import Net, decoder, vgg
import os
import wandb
import random
import torch.nn as nn

def init_wandb(cfg):
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    project_name = 'SEG2MR' + str(random.randint(1,1000))
    wandb.init(project=project_name)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def seed(seed_value):
    if seed_value == -1:
        return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def print_infor(cfg, dataloader):
    cfg.epoch_size = cfg.iters // len(dataloader)
    cfg.print_cfg() 

    print('\n===========================================================')
    print('Dataloader Ok!')
    print('-----------------------------------------------------------')
    print('[Data Size]:',len(dataloader.dataset))
    print('[Batch Size]:',cfg.batch_size)
    print('[One epoch]:',len(dataloader.dataset)//cfg.batch_size,'step   # (Data Size / Batch Size)')
    print('[Epoch & Iteration]:',cfg.epoch_size,'epoch &', cfg.iters,'step')
    print('-----------------------------------------------------------')
    print('===========================================================')


def def_model(cfg):
    vgg_net = vgg
    vgg_net.load_state_dict(torch.load(cfg.vgg_path))
    vgg_net = nn.Sequential(*list(vgg_net.children())[:31])
    model = Net(vgg_net, decoder)
    model.train().cuda()
    return model


def def_optim(cfg, model):
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return optim


def load_model_optim(cfg, model, optim):
    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume)['model'])
        optim.load_state_dict(torch.load(cfg.resume)['optim'])


def load_iter_epoch(cfg, dataloader):
    if cfg.resume:
        iter = torch.load(cfg.resume)['iter']
        epoch = int(iter/len(dataloader)) 
    else:
        iter = 0
        epoch = 0
    return iter, epoch
