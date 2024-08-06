import argparse
from config import update_config
from train_folder.train_pre_func import * 
from train_folder.train_func import Trainer
from dataset import make_loader, make_split


def main():
    parser = argparse.ArgumentParser(description='seg2mr')
    parser.add_argument('--wandb', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--iters', default=100000, type=int, help='The total iteration number.')
    parser.add_argument('--resume', default=None, type=str, help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
    parser.add_argument('--val_interval', default=1000, type=int)
    parser.add_argument('--save_interval', default=5000, type=int)
    parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
    parser.add_argument('--style', default='star', type=str)

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Pre-work for Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # init wandb
    if train_cfg.wandb:
        init_wandb(train_cfg)
    
    # setup seed (for deterministic behavior)
    seed(seed_value=train_cfg.manualseed)

    # train/test data split
    make_split(train_cfg)

    # get dataset and loader
    train_loader = make_loader(train_cfg, train=True, batch_size=train_cfg.batch_size)
    print_infor(cfg=train_cfg, dataloader=train_loader)

    # define model
    model = def_model(train_cfg)

    # define optimizer 
    optim = def_optim(train_cfg, model)

    # load model, optim
    load_model_optim(train_cfg, model, optim)

    # load iter, epoch
    iter, epoch = load_iter_epoch(train_cfg, train_loader)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # train
    Trainer(train_cfg, train_loader, model, optim, iter, epoch, device)


if __name__=="__main__":
    main()