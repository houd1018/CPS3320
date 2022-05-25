# CPS3320
This repository is for 2022SP CPS3320 python.
## Group member
Dazhou Hou(Howard)

Jiayi Zhang(Libre)

## [tutorials](https://github.com/houd1018/CPS3320/tree/master/tutorials)

This directory contains in-class tutorials

## [FewShot_Siamese](https://github.com/houd1018/CPS3320/tree/master/FewShot_Siamese)

The final project for 2022SP CPS3320

### Model Architecture

![图片1](https://github.com/houd1018/CPS3320/blob/master/resource/%E5%9B%BE%E7%89%871.png)

### [sourceCode_PyTorch](https://github.com/houd1018/CPS3320/tree/master/FewShot_Siamese/sourceCode_PyTorch)

Train the model

```
python train.py
```

- <u>You can train your model from scratch or use pre-trained model.</u>

Predict the test image

```
python predict.py
```

- <u>Model file must be assigned</u>

- <u>You have to specify test_image and support_dataset</u>

### [Hyperparameters](https://github.com/houd1018/CPS3320/tree/master/FewShot_Siamese/sourceCode_PyTorch/runs)

### Data Processing

####  [Data Augmentation](https://github.com/houd1018/CPS3320/blob/master/FewShot_Siamese/sourceCode_PyTorch/utils/dataloader.py)

- flipping
- spinning
- cropping
- Hue

#### [image_scraping](https://github.com/houd1018/CPS3320/tree/master/FewShot_Siamese/image_scraping)

Use selenium to scrape images from Google Image

### Result

### Training Record

#### accuracy

#### loss

#### learning rate

### Reference website

- Reconstruct Dataloader
  - [DATASETS & DATALOADERS tutorial](l https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - [DATASETS & DATALOADERS doc](l https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)
  - [Data Augmentation Compilation with Python and OpenCV](l https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0)
- VGG-16
  - [VGG-16: A simple implementation using Pytorch](l https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1)
- Image Scraping
  - [How To Web Scrape & Download Images With Python](l https://www.youtube.com/watch?v=NBuED2PivbY)
- Papers related to Few-shot Learning & Siamese Neural Network
  - [One-shot learning of object categories](l https://ieeexplore.ieee.org/abstract/document/1597116/)
  - [Signature Verification using a "Siamese" Time Delay Neural Network](l https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html)
  - [Siamese Neural Networks for One-Shot Image Recognition](l http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)

