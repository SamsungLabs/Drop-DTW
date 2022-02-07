import sys
import os

# MIL-NCE code
S3D_PATH = '/user/n.dvornik/Git/S3D_HowTo100M/'
# paths to datasets
COIN_PATH = '/user/n.dvornik/Datasets/COIN/'
YC_PATH, CT_PATH = None, None

# root project and weights folder
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'weights')