import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.siamese import Siamese as siamese

'''
This class is used for predicting

'''

class Siamese(object):
    _defaults = {

        # put your model path(logs folder) here
        # if use this model to predict
        "model_path"    : 'logs\ep100-loss0.003-val_loss1.042.pth',

        # image size
        "input_shape"   : (105, 105, 3),

        # GPU
        "cuda"          : True
    }

    # check default settings
    # @classmethod: no need for self and instance
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # constructor
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()


    def generate(self):
        # load model and weights
        print('Loading weights into state dict...')
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model   = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    # preprocessing: resize the image funtion
    def letterbox_image(self, image, size):
        image   = image.convert("RGB")
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
        
   
    # Predict: take two image as input
    def detect_image(self, image_1, file_list):
        # -------------test image process-------------------
        
        # resize image
        image_1 = self.letterbox_image(image_1,[self.input_shape[1],self.input_shape[0]])
        
        # normalization
        photo_1 = np.asarray(image_1).astype(np.float64) / 255

        # make sure it is RGB(3 channels)
        if self.input_shape[-1]==1:
            photo_1 = np.expand_dims(photo_1, -1)

        with torch.no_grad():
            # add one more dimension(the batch dimention)
            # then put it into network correctly
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            
            if self.cuda:
                photo_1 = photo_1.cuda()

        # -------------plotitng----------------
        plt.subplot(2, len(file_list), len(file_list) // 2)
        plt.imshow(np.array(image_1))

        # ---------------------support dataset------------------------
        for idx, image_2 in enumerate(file_list):
            image_2 = Image.open(image_2)
            # resize image
            image_2 = self.letterbox_image(image_2,[self.input_shape[1],self.input_shape[0]])
            
            # normalization
            photo_2 = np.asarray(image_2).astype(np.float64) / 255

            # make sure it is RGB(3 channels)
            if self.input_shape[-1]==1:
                photo_2 = np.expand_dims(photo_2, -1)

            with torch.no_grad():
                # add one more dimension(the batch dimention)
                # then put it into network correctly
                photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)
                
                if self.cuda:
                    photo_2 = photo_2.cuda()

                # get similarity result
                output = self.net([photo_1, photo_2])[0]
                # add sigmoid here to ouput similarity between 0-1
                output = torch.nn.Sigmoid()(output)

            # --------------plotting---------------
            plt.subplot(2, len(file_list), len(file_list) + idx + 1)
            plt.imshow(np.array(image_2))
            plt.text(20, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        plt.show()
