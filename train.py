import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import samples.field.field as field
import mrcnn.utils as utils
import mrcnn.visualize as visualize
import mrcnn.model as modellib
from mrcnn.model import log
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = field.FieldConFig()
config.USE_MINI_MASK = False

dataset = field.FieldDataset()
dataset.load_field()
dataset.prepare()

config.display()
model = modellib.MaskRCNN(mode="training", config=config, model_dir="./logs")

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

model.train(dataset, dataset, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')