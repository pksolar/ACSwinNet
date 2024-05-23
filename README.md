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


## Installation

Set up your environment:

```bash
 pip install -r requirements.txt
```
## Usage
1. Train  DAE model. To obtain the weight files of the segmentation image encoder and the original image encoder by training the DAE model, the command line is:
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
3. Train  ACSwinNet model. 
```bash
 python ./train.py --train_images_file dataset/train/*.nii.gz --val_images_file dataset/val/*.nii.gz --autoencoder_file_seg pth/encoder_seg.pth --autoencoder_file_image pth/encoder_image.pth  --results_dir result/ --ref_img_path path/to/ref_img
```
4. Infer  ACSwinNet model.
```bash
 python ./infer.py --test_images_file dataset/test/*.nii.gz --model_file pth   --results_dir result_test/ --ref_img_path path/to/ref_img
```
## Reference
我在我的项目中借鉴了 [JohnDoe 的代码仓库](https://github.com/JohnDoe/awesome-project)，并进行了适当的修改和使用。
