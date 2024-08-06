import torch 
import matplotlib.pyplot as plt


def make_models_dict(model, opt, iter):
    model_dict = {'model': model.state_dict(), 
                  'optim': opt.state_dict(),
                  'iter': iter}
    return model_dict


def update_best_model(cfg, iter, curr_loss, best_loss, model, opt):
    if curr_loss < best_loss:
        best_loss = curr_loss
        model_dict = make_models_dict(model, opt, iter+1)
        save_path = f"{cfg.weights_path}/best_model.pth"
        torch.save(model_dict, save_path)

        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f"[best model] update! at {iter+1} iteration!! [loss: {curr_loss:.3f}]")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    return best_loss