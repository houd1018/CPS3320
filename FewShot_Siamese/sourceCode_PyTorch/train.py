'''
Entire Training process
I have a template for other deep learning project, thus I modified some and transfered it to our project
Mostly coded by ourselves
'''


import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.siamese import Siamese
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils_fit import fit_one_epoch
from torch.utils.tensorboard import SummaryWriter

# get the total number of images
def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'images')
        for character in os.listdir(train_path):
            # count the number of images in subfolders
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    # train for Omniglot dataset
    else:
        train_path = os.path.join(path, 'images')
        for alphabet in os.listdir(train_path):
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num

if __name__ == "__main__":
    
    writer = SummaryWriter()

    Cuda            = True
  
    dataset_path    = "./datasets"
   
    input_shape     = [105,105,3]

    # train your own dataset: True
    # train Omniglot dataset: False
    train_own_data  = True

    # if you have pretrained model(VGG-16), put it in the model_data folder
    pretrained      = True
    
    # this model path(logs) is for contiue to train, not pretrained model
    model_path      = ""
    
    # initiate the model
    model = Siamese(input_shape, pretrained)

    # keep training for break point
    if model_path != '':
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # set to train mode
    # use GPU
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss = nn.BCELoss()

    # split validation and training dataset
    train_ratio         = 0.9
    images_num          = get_image_num(dataset_path, train_own_data)
    num_train           = int(images_num * train_ratio)
    num_val             = images_num - num_train
    
    # two stage training, initial learning rate is different
    #---------------------------first training stage----------------------------------#
    if True:
        Batch_size      = 32
        Lr              = 1e-4
        Init_epoch      = 0
        Freeze_epoch    = 50

        epoch_step          = num_train // Batch_size
        epoch_step_val      = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('the size of dataset is not enough')
        
        # Adam: momentum & adaptive learning rate
        optimizer       = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)
        
        # set up dataset
        train_dataset   = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True, train_own_data=train_own_data)
        val_dataset     = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False, train_own_data=train_own_data)

        # set up dataloader
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True, 
                                drop_last=True, collate_fn=dataset_collate)
        global_epoch = 0

        # ----------------training-----------------------
        for epoch in range(Init_epoch, Freeze_epoch):
            # one epoch
            total_loss, val_loss, total_accuracy, val_total_accuracy, lr = fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Freeze_epoch, Cuda)

            # gradient decent
            lr_scheduler.step()
            
            # keep tracking  
            writer.add_scalar("Total Loss", total_loss, global_step = global_epoch)
            writer.add_scalar("Val Loss", val_loss, global_step = global_epoch)
            writer.add_scalar("Total accuracy", total_accuracy, global_step = global_epoch)
            writer.add_scalar("Val accuracy", val_total_accuracy, global_step = global_epoch)            
            writer.add_scalar("Learning rate", lr, global_step = global_epoch)
            global_epoch = global_epoch + 1

    
    #---------------------------second training stage----------------------------------#
    if True:
        Batch_size      = 32
        Lr              = 1e-5
        Freeze_epoch    = 50
        Unfreeze_epoch  = 100

        epoch_step          = num_train // Batch_size
        epoch_step_val      = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('the size of dataset is not enough')

        optimizer       = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

        train_dataset   = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True, train_own_data=train_own_data)
        val_dataset     = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False, train_own_data=train_own_data)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Freeze_epoch, Unfreeze_epoch):
            total_loss, val_loss, total_accuracy, val_total_accuracy, lr = fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Freeze_epoch, Cuda)
            lr_scheduler.step()            
            writer.add_scalar("Total Loss", total_loss, global_step = global_epoch)
            writer.add_scalar("Val Loss", val_loss, global_step = global_epoch)
            writer.add_scalar("Total accuracy", total_accuracy, global_step = global_epoch)
            writer.add_scalar("Val accuracy", val_total_accuracy, global_step = global_epoch)            
            writer.add_scalar("Learning rate", lr, global_step = global_epoch)
            global_epoch = global_epoch + 1
