import os
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
