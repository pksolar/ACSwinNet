# ACSwinNet: A Deep Learning-Based Rigid Registration Method for Head-Neck CT-CBCT Images in Image-Guided Radiotherapy


## Introduction

By integrating a hybrid architecture that combines SwinTransformer with traditional Convolutional Neural Networks (CNNs),
we have enhanced the registration weight for the head region in the CBCT-CT rigid registration task, 
aiming to achieve more precise alignment of the head area. Paper download address:
## Framework
The overview of ACSwinNet:

![image](https://github.com/pksolar/ACSwinNet/assets/37496977/22612ea2-8105-43fa-894a-40fb19ee69ea)

------------------------------------------------------------------------------------------------------

Architecture of DAE:

![image](https://github.com/pksolar/ACSwinNet/assets/37496977/7aa14cb7-becc-4af8-afab-e232e0e9f422)

------------------------------------------------------------------------------------------------------

Metric Network:

![image](https://github.com/pksolar/ACSwinNet/assets/37496977/de42ed0d-0560-46f9-97cf-f343f7c8e4db)

## Operating Environment:

Operating System: Windows 10
CPU: i9-10900k
RAM: 32GB
GPU: Nvidia Geforce RTX 3090(24GB)

## Installation

Set up your environment:

```bash
 pip install -r requirements.txt
```
## Usage
1. Train  DAE model. To obtain the weight files of the segmentation image encoder and the original image encoder by training the DAE model. the command line is:
```bash
 python ./DAE_model/train.py --image_type image --train_images_file path/to/train_images    --save_model_path path/to/save_model --ref_img_path path/to/ref_img
```
```bash
 python ./DAE_model/train.py --image_type seg --train_seg_file path/to/train_seg   --save_model_path path/to/save_model --ref_img_path path/to/ref_img
```
2. Infer  DAE model.To assess the denoising effect by performing inference on the DAE model：
```bash
 python ./DAE_model/infer.py --image_type image --test_images_file path/to/test_images  --save_model_path path/to/save_model --ref_img_path path/to/ref_img
```
```bash
 python ./DAE_model/infer.py --image_type seg --test_images_file path/to/test_seg  --save_model_path path/to/save_model --ref_img_path path/to/ref_img
```
3. Train  ACSwinNet model. The _encoder_image.pth_ file is utilized by the metric network, while the _encoder_seg.pth_ file is employed to encode the segmentations.
```bash
 python ./train.py --train_images_file dataset/train/*.nii.gz --val_images_file dataset/val/*.nii.gz --autoencoder_file_seg pth/encoder_seg.pth --autoencoder_file_image pth/encoder_image.pth  --results_dir result/ --ref_img_path path/to/ref_img
```
4. Infer  ACSwinNet model.
```bash
 python ./infer.py --test_images_file dataset/test/*.nii.gz --model_file pth   --results_dir result_test/ --ref_img_path path/to/ref_img
```
## Data
We are currently in the process of applying to make the dataset public. Once the application is approved, we will promptly update the dataset in the code repository.

## Acknowledgment

We have referenced Lucas Mansilla's code([https://github.com/JohnDoe/awesome-project](https://github.com/lucasmansilla/ACRN_Chest_X-ray_IA)) 
Guha Balakrishnan's code(http://voxelmorph.csail.mit.edu),
Liu's code(https://github.com/microsoft/Swin-Transformer)
in my project, making appropriate modifications and utilization.
Thank you for the contributions of these authors.


