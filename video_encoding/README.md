## Handling COIN 
Here we cover how to download the COIN dataset and extract the features from it, using the S3D network pretrained on HowTo100M [1].

### Step-by-step instruction
1. Download the COIN dataset on your machine. Follow the instruction on the [official web site](https://coin-dataset.github.io/).
2. Specify the path to the COIN dataset in `path.py` file (in the root directory), i.e. set COIN_PATH = <location of COIN on you machine>.
3. Run `video_encoding/setup_COIN.py`. It will download the videos from youtube, extract features and pack them into an lmdb dataset.

[1] - Miech et al. "End-to-end learning of visual representations from uncurated instructional videos." CVPR'20.