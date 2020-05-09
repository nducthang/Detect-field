import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

class FieldConFig(Config):
    NAME = "field"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # Background + field
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class FieldDataset(utils.Dataset):
    def load_field(self):
        self.add_class("field", 1, "field")
        dataset_image = "./data/field/train"
        onlyfiles = [f for f in listdir(dataset_image) if isfile(join(dataset_image, f))]
        print("Num files:", len(onlyfiles))
        for i in range(len(onlyfiles)):
            self.add_image(
                "field", 
                image_id = int(onlyfiles[i][:-4]),
                path=None, 
                width=2048, 
                height=2048)
    
    def load_image(self, image_id):
        dataset_image = "./data/field/train"
        image = mpimg.imread(os.path.join(dataset_image, str(image_id) + '.jpg'))
        return image


    def load_mask(self, image_id):
        dataset_mask = "./data/field/mask"
        mask = np.load(os.path.join(dataset_mask, str(image_id) + '.npy'))
        class_ids = np.ones(mask.shape[2], dtype=np.int32)
        return mask, class_ids