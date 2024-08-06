import torch 
import random
from eval import val
from train_folder.train_ing_func import *
from train_folder.show_gpu import showGPU
import time
import datetime
import math
import wandb
from dataset import img2tensor


class Trainer:
    def __init__(self, cfg, dataloader, model, opt, iter, epoch, device):
        self.dataloader = dataloader
        self.model = model
        self.opt = opt
        self.iters = cfg.iters
        self.device = device
        self.wandb = cfg.wandb

        # get styles
        self.style = img2tensor(path=cfg.style_path) # (1, c, h, w)

        # train init
        self.start_iter = iter
        self.iter_num = iter
        self.epoch_num = epoch
        self.curr_loss = math.inf
        self.best_loss = math.inf
        self.Training = True  

        # training start
        self.fit(cfg)


    def save_model(self, cfg):
        model_dict = make_models_dict(self.model, self.opt, self.iter_num+1)
        save_path = f"{cfg.weights_path}/{self.iter_num+1}.pth"
        torch.save(model_dict, save_path)
        print(f"\nAlready saved: \'{save_path}'.")


    def one_time(self, iter_num, temp, start_iter):
        torch.cuda.synchronize()
        time_end = time.time()
        if iter_num > start_iter:  
            iter_t = time_end - temp
        else:
            iter_t = None
        temp = time_end
        return iter_t, temp


    def one_print(self, pr_loss, pr_loss_c, pr_loss_s, iter_t, eta):
        print(f"\n[{self.iter_num + 1}/{self.iters}] total_loss: {pr_loss:.3f} | content_loss: {pr_loss_c:.3f} | style_loss: {pr_loss_s:.3f} | "\
              f"iter_t: {iter_t:.3f} | remain_t: {eta} | best_loss: {self.best_loss:.3f}")
        if self.wandb:
            wandb.log({"total_loss": pr_loss.item()})
            wandb.log({"content_loss": pr_loss_c.item()})
            wandb.log({"style_loss": pr_loss_s.item()})


    def one_forward(self, cfg, contents, styles):
        _, loss_c, loss_s = self.model(content=contents, style=styles)
        loss_c = cfg.content_weight * loss_c
        loss_s = cfg.style_weight * loss_s
        loss = loss_c + loss_s
        return loss, loss_c, loss_s


    def fit(self, cfg):
        print('\n===========================================================')
        print('Training Start!')
        print('===========================================================')

        # prepare
        time_temp = 0
        total_loss = 0
        total_loss_c = 0
        total_loss_s = 0 

        # show GPU
        showGPU()

        while self.Training:
            # one epoch
            for contents in self.dataloader:
                contents = contents[0].cuda()
                batch_size = contents.shape[0]
                styles = self.style.repeat(batch_size, 1, 1, 1).cuda() # (b, c, h, w)

                # calculate time
                if self.iter_num > 0:
                    try:
                        iter_t, time_temp = self.one_time(self.iter_num, time_temp, self.start_iter)
                        time_remain = (self.iters - self.iter_num) * iter_t
                        eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    except:
                        eta = '?'

                # forward
                loss, loss_c, loss_s = self.one_forward(cfg, contents, styles)
                total_loss += loss
                total_loss_c += loss_c
                total_loss_s += loss_s

                if (self.iter_num + 1) % 20 == 0:
                    pr_loss = total_loss / (self.iter_num % cfg.val_interval)
                    pr_loss_c = total_loss_c / (self.iter_num % cfg.val_interval)
                    pr_loss_s = total_loss_s / (self.iter_num % cfg.val_interval)
                    self.one_print(pr_loss, pr_loss_c, pr_loss_s, iter_t, eta)

                # optimization
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # val model
                if (self.iter_num+1) % cfg.val_interval == 0:
                    self.curr_loss = total_loss / cfg.val_interval
                    total_loss = 0
                    total_loss_c = 0
                    total_loss_s = 0 
                    self.best_loss = update_best_model(cfg, self.iter_num, self.curr_loss, self.best_loss, self.model, self.opt)
                    val(cfg, self.model, self.iter_num)

                # save model
                if (self.iter_num+1) % cfg.save_interval == 0:
                    self.save_model(cfg)
                    
                # training end
                if (self.iter_num+1) == self.iters:
                    self.Training = False
                    break
                
                # update iter_num
                self.iter_num += 1

            # update epoch_num
            self.epoch_num += 1

