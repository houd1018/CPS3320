
'''
This file is coded by ourselves
'''

'''
This class is used for training
set up Siamese network
compare how similar two images are

'''

import torch
import torch.nn as nn

from nets.vgg import VGG16

# get the length in one dimension after convolution
def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()

        # connect to VGG16
        self.vgg = VGG16(pretrained, input_shape[-1])

        # remove original avgpool and classifier, because this is not a classification task
        del self.vgg.avgpool
        del self.vgg.classifier
        
        # flatten
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])

        # two fully connected layers to make sure we get a vector whose length is 1
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x

        #   put to inputs into VGG-16 to get features
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2) 
          

        #  get L-1 distance (take abosulute value and substract)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)

    
        # fully connected twice
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)

        # put sigmoid later
        return x
