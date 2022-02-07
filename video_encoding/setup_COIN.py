#!/usr/bin/env python3
import os
import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import COIN_PATH, PROJECT_PATH

################# Handling MIL-NCE S3D network #################
weights_path = osp.join(PROJECT_PATH, 'video_encoding', 'model_weights')
if not osp.isdir(weights_path):
    os.mkdir(weights_path)
os.system(f"wget wget -P {weights_path} https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth")
os.system(f"wget wget -P {weights_path} https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy")

################## Handling the COIN dataset #################
# setting up folders
coin_tfrec = osp.join(COIN_PATH, "videos_tfrecords")
coin_lmdb = osp.join(COIN_PATH, "lmdb")
for folder in [coin_tfrec, coin_lmdb]:
    if not osp.isdir(folder):
        os.mkdir(folder)
tfrec_script_path = osp.join(PROJECT_PATH, 'video_encoding', 'InstVids2TFRecord_COIN.py')
lmdb_script_path = osp.join(PROJECT_PATH, 'video_encoding', 'encode_lmdb.py')

# write videos into TFRecords for each split of COIN
for mode in ['train', 'val']:
    os.system(f"python3 {tfrec_script_path} --mode={mode}")

# encode TFRecords with S3D into lmdb feature datasets
os.system(f"python3 {lmdb_script_path} --source {coin_tfrec} --dest {coin_lmdb} --dataset COIN")
