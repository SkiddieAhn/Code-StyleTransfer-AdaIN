from glob import glob
import os


if not os.path.exists('weights'):
    os.mkdir('weights')
if not os.path.exists('results'):
    os.mkdir('results')

share_config = {'mode': 'training',
                'data_path': 'data/img_align_celeba'}  

class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    share_config['style_weight'] = 10.0
    share_config['content_weight'] = 1.0
    share_config['vgg_path'] = 'pretrained/vgg.pth'
    share_config['style'] = args.style
    share_config['style_path'] = os.path.join('data', f'{args.style}.jpg')
    share_config['weights_path'] = os.path.join('weights', f'{args.style}')

    if mode == 'train':
        share_config['wandb'] = args.wandb
        share_config['batch_size'] = args.batch_size
        share_config['lr'] = 0.0001
        share_config['iters'] = args.iters
        share_config['resume'] = glob(share_config['weights_path']+f'/{args.resume}*')[0] if args.resume else None
        share_config['val_interval'] = args.val_interval
        share_config['save_interval'] = args.save_interval
        share_config['manualseed'] = args.manualseed
        share_config['results_path'] = os.path.join('results', f'{args.style}')

        # make weights directory 
        if not os.path.exists(share_config['weights_path']):
            os.makedirs(share_config['weights_path'], exist_ok=True)

        # make results directory
        if not os.path.exists(share_config['results_path']):
            os.makedirs(share_config['results_path'], exist_ok=True)

    elif mode == 'test':
        share_config['trained_model'] = args.trained_model
        share_config['content'] = args.content
        share_config['content_path'] = os.path.join('data', f'{args.content}.jpg')
        share_config['test_name'] = args.content + '_' + args.style
        share_config['test_path'] = os.path.join('results', share_config['test_name'])

        # make test results directory
        if not os.path.exists(share_config['test_path']):
            os.makedirs(share_config['test_path'], exist_ok=True)


    return dict2class(share_config)  # change dict keys to class attributes