
'''
Coded by ourselves
'''

import os
import numpy as np
from PIL import Image

from predict_siamese import Siamese

# get support dataset's information in the list
def get_allfile(path):  
    all_file = []
    for f in os.listdir(path): 
        f_name = os.path.join(path, f)
        all_file.append(f_name)
    return all_file

if __name__ == "__main__":
    model = Siamese()
        
    while True:
        image_test = input('Input test image filename:')
        try:
            image_test = Image.open(image_test)
        except:
            print('Test image Open Error! Try again!')
            continue

        support_path = input('Input support dataset path:')
        try:
            all_file = get_allfile(support_path)
        except:
            print('Support dataset Open Error! Try again!')
            continue
        model.detect_image(image_test,all_file)
