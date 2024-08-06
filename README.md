# Adaptive Instance Normalization
Pytorch implementation of style transfer paper for ICCV 2017: [Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868).  
Most codes were obtained from the following GitHub page: [[Link]](https://github.com/naoto0804/pytorch-AdaIN)  

### The network pipeline.  
![adain_pipe](https://github.com/user-attachments/assets/df55c1a1-a07a-4c05-bfc3-bdab1a4aed08)

## Environments  
PyTorch >= 1.1.  
Python >= 3.6.  
torchvision  
pillow  
wandb  
Other common packages.  

## Dataset
- **Celeb-A Faces** dataset is a large collection of celebrity facial images used for tasks like facial recognition and generative modeling.  
- It includes over ```200,000``` diverse celebrity images.  
- We trained this dataset as the content images for AdaIN.
- Please download the data from the Google Drive link below, unzip the zip file, and then move it to the ```data``` directory.

|     Celeb-A Faces            |
|:------------------------:|
| [Google Drive](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)   |

## Training
- Please verify the training data path in ```config.py```.
- When training starts, the ```weights/{style_name}```, and ```results/{style_name}``` folders are automatically created.
- All saved models are located in the ```weights/{style_name}``` folder.
- You can save your desired style image in the ```data``` folder and then proceed with training.
  
```Shell
# recommended option for training
python train.py --wandb=True --batch_size=64 --iters=30000 --val_interval=100 --save_interval=5000 --style=star
# Continue training with latest model
python train.py --wandb=True --batch_size=64 --iters=30000 --val_interval=100 --save_interval=5000 --style=star --resume=5000
```

## Testing
- You can save your desired content image in the ```data``` folder and then proceed with testing.
- You can save your desired style image in the ```data``` folder and then proceed with testing.
  
```Shell
# testing example
python eval.py --trained_model=best_model --content=people --style=star
```

## Pre-trained Models
- Download the encoder weight (vgg) and put it under the ```pretrained``` folder.
- Download the decoder weight (style) and put it under the ```weights/{style_name}``` folder.

|   VGG     | Style: Sad    | Style: Star    |
|:--------------:|:-----------:|:-----------:|
|[Google Drive](https://drive.google.com/file/d/1ZiQWBvXwOBNOtPc9J0PMKAdXTDGQq7GR/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1V5tr6B_6Hx8KDdSvOq7szwg4qwy9Otbi/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1wJNlJhHUS4ESSm6lT_YJ90ZvD6m01Mfo/view?usp=drive_link)|

## Results
#### Validation results can be found on the path ```results/{style_name}```.  
| Style: Sad                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
![sad](https://github.com/user-attachments/assets/d9a6c40e-e31e-4560-ac54-bd1e37a25d52)

| Style: Star                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
![star](https://github.com/user-attachments/assets/9598612b-a1a8-4e28-aecc-a8604a55f661)

#### Test results can be found on the path ```results/{content_name}_{style_name}```.  
| Content: Changseop Kim, Style: Sad                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
![csad](https://github.com/user-attachments/assets/16e524dc-a960-4be2-be5d-104a39759b2b)

| Content: Changseop Kim, Style: Star                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
![cstar](https://github.com/user-attachments/assets/83c8c8e5-419c-41d2-b78e-5ba6e1862d85)

