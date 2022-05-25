'''
Some part is from Internet, such as mannual data augumentation which related to cv2
Most of code is coded by ourselves
'''


import os
import random
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

'''
the way how Siamese works is different from default training
it will choose two same kind of images as positive pair, and label them to 1
then choose two different kinds of images as negative pair, and laebel them to 0
'''
'''
this file is used to redesign the dataloader which is based on PyTorch default dataloader
'''

# partial random
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# extends Dataset
class SiameseDataset(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_val, train=True, train_own_data=True):
        super(SiameseDataset, self).__init__()

        self.dataset_path   = dataset_path
        self.image_height   = input_shape[0]
        self.image_width    = input_shape[1]
        self.channel        = input_shape[2]
        
        # path & label
        self.train_lines    = []
        self.train_labels   = []
   
        self.val_lines      = []
        self.val_labels     = []
        self.types          = 0

        self.num_train      = num_train
        self.num_val        = num_val
        self.train          = train
        self.train_own_data = train_own_data
        self.load_dataset()

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_val

    def load_dataset(self):

        # count types, labers, path
        train_path = os.path.join(self.dataset_path, 'images')
        if self.train_own_data:
            for character in os.listdir(train_path):

                character_path = os.path.join(train_path, character)
                for image in os.listdir(character_path):
                    self.train_lines.append(os.path.join(character_path, image))
                    self.train_labels.append(self.types)
                self.types += 1
        else:
            for alphabet in os.listdir(train_path):
                alphabet_path = os.path.join(train_path, alphabet)
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    for image in os.listdir(character_path):
                        self.train_lines.append(os.path.join(character_path, image))
                        self.train_labels.append(self.types)
                    self.types += 1

        # ------------------shuffle----------------
        random.seed(1)
        shuffle_index = np.arange(len(self.train_lines), dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        self.train_lines    = np.array(self.train_lines,dtype=np.object)
        self.train_labels   = np.array(self.train_labels)
        self.train_lines    = self.train_lines[shuffle_index]
        self.train_labels   = self.train_labels[shuffle_index]
        
        # ------------------split val & train------------
        self.val_lines      = self.train_lines[self.num_train:]
        self.val_labels     = self.train_labels[self.num_train:]
    
        self.train_lines    = self.train_lines[:self.num_train]
        self.train_labels   = self.train_labels[:self.num_train]

    # Data Augmentation
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, flip_signal=False):
        if self.channel == 1:
            image = image.convert("RGB")

        h, w = input_shape


        # ----------------cropping-------------
        rand_jit1   = rand(1-jitter,1+jitter)
        rand_jit2   = rand(1-jitter,1+jitter)
        new_ar      = w/h * rand_jit1/rand_jit2

        scale = rand(0.75,1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        

        # -----------flipping-----------------
        flip = rand()<.5
        if flip and flip_signal: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # ----------pasting-----------------
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (255,255,255))

        new_image.paste(image, (dx, dy))
        image = new_image

        # ---------spining------------
        rotate = rand()<.5
        if rotate: 
            angle = np.random.randint(-5,5)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue = [255,255,255]) 

        # ---------Hue-------------
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        return image_data

# path to images and label which are formatted for the dataloader
    def _convert_path_list_to_images_and_labels(self, path_list):

        #   if batch_size = 16
        #   len(path_list)/2 = 32
        number_of_pairs = int(len(path_list) / 2)

        # define input images and labels
        pairs_of_images = [np.zeros((number_of_pairs, self.channel, self.image_height, self.image_width)) for i in range(2)]
        labels          = np.zeros((number_of_pairs, 1))



        #  0 is different type
        #  1 is same type
        for pair in range(number_of_pairs):
            # -----------------1---------------------
            image = Image.open(path_list[pair * 2])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            image = np.transpose(image, [2, 0, 1])
            image = image / 255
            if self.channel == 1:
                pairs_of_images[0][pair, 0, :, :] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image


            # -----------------0-----------------------
            image = Image.open(path_list[pair * 2 + 1])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            image = np.transpose(image, [2, 0, 1])
            image = image / 255
            if self.channel == 1:
                pairs_of_images[1][pair, 0, :, :] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        
        # -------------shuffle combination------------------
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        return pairs_of_images, labels


    # overwrite __getitem___
    def __getitem__(self, index):
        if self.train:
            lines = self.train_lines
            labels = self.train_labels
        else:
            lines = self.val_lines
            labels = self.val_labels
    
        batch_images_path = []
 

        # randomly choose three images
        # two is in the same type, the other is different
        c               = random.randint(0, self.types - 1)
        selected_path   = lines[labels[:] == c]
        while len(selected_path)<3:
            c               = random.randint(0, self.types - 1)
            selected_path   = lines[labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)


        # 0 and 1 is same, so the network is supposed to give 1
        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        # 2 and 1 is different, so the network is supposed to give 0
        batch_images_path.append(selected_path[image_indexes[2]])

        # process the different image
        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = lines[labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = lines[labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])
        
        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels

# collate lists of samples into batches
def dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    return np.array([left_images, right_images]), np.array(labels)
