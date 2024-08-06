import torch 
import argparse
from train_folder.train_pre_func import str2bool 
from eval_folder.train_eval import val_train_eval
from eval_folder.test_eval import val_test_eval
from config import update_config
from model.net import Net, decoder, vgg
import torch.nn as nn


parser = argparse.ArgumentParser(description='Contrastive_learning')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--content', default='people', type=str)
parser.add_argument('--style', default='star', type=str)

def val(cfg, model=None, iter=None):
    '''
    ========================================
    This is for evaluation during training.    
    ========================================
    '''
    if model:
        model.eval()
        val_train_eval(cfg, model, iter+1)
        model.train()
        return


    '''
    ========================================
    This is for evaluation during testing.    
    ========================================
    '''
    vgg_net = vgg
    vgg_net.load_state_dict(torch.load(cfg.vgg_path))
    vgg_net = nn.Sequential(*list(vgg_net.children())[:31]) 

    model = Net(vgg_net, decoder)
    model.eval().cuda()

    if cfg.trained_model:
        model.load_state_dict(torch.load(f'{cfg.weights_path}/' + cfg.trained_model + '.pth')['model'])
        iter = torch.load(f'{cfg.weights_path}/' + cfg.trained_model + '.pth')['iter']
        val_test_eval(cfg, model, iter)
    else:
        print('no trained model!')


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
