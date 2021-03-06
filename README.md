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

![output](https://github.com/houd1018/CPS3320/blob/master/resource/output.png)

### Training Record

#### accuracy

![Total-accuracy](https://github.com/houd1018/CPS3320/blob/master/resource/Total-accuracy.jpg)

#### loss

![Total-loss](https://github.com/houd1018/CPS3320/blob/master/resource/Total-Loss.jpg)

#### learning rate

![Learning-rate](https://github.com/houd1018/CPS3320/blob/master/resource/Learning-rate.jpg)

### [Download Dataset](https://drive.google.com/file/d/1LhxNkVzmsMFYYnZEHduBMbb4rIbjZG2W/view?usp=sharing)

### Download Weight File

- [VGG-16 pretrained model](https://drive.google.com/file/d/15zyOhp_RzO4r5494G_0isPmjXfJNPhOR/view?usp=sharing)
- [animal-5 classification model](https://drive.google.com/file/d/1rwu9e264xYzptNfgCuZkzD0C9lVyZuu9/view?usp=sharing)

### Reference website

- Reconstruct Dataloader
  - [DATASETS & DATALOADERS tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - [DATASETS & DATALOADERS doc](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)
  - [Data Augmentation Compilation with Python and OpenCV](https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0)
- VGG-16
  - [VGG-16: A simple implementation using Pytorch](https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1)
- Image Scraping
  - [How To Web Scrape & Download Images With Python](https://www.youtube.com/watch?v=NBuED2PivbY)
- Papers related to Few-shot Learning & Siamese Neural Network
  - [One-shot learning of object categories](https://ieeexplore.ieee.org/abstract/document/1597116/)
  - [Signature Verification using a "Siamese" Time Delay Neural Network](https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html)
  - [Siamese Neural Networks for One-Shot Image Recognition](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)

