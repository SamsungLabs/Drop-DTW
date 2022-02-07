## Pre-requisities: run 'pip install youtube-dl' to install the youtube-dl package.

import os
import sys
import json
import os.path as osp
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import COIN_PATH

output_path = osp.join(COIN_PATH, 'videos')
json_path = osp.join(COIN_PATH, 'COIN.json')
DL_VID = True # download video flag
DL_NAR = False # download subtitles flag

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
data = json.load(open(json_path, 'r'))['database']
youtube_ids = list(data.keys())
print(len(youtube_ids))
for youtube_id in tqdm(data):
    info = data[youtube_id]
    task_class = info['class']
    subset = info['subset']
    annotation = info['annotation']
    typer = info['recipe_type']
    url = info['video_url']
    st = info['start']
    ed = info['end']
    duration = info['duration']
    
    mode = 'train' if 'train' in subset else 'val'
    vid_loc = output_path + '/' + mode + '/' + task_class + '_' + str(typer)
    # print(vid_loc)
    # print(url, st, ed, duration)
    if not os.path.exists(vid_loc):
        os.makedirs(vid_loc)
    if DL_VID:
        os.system('youtube-dl -o ' + vid_loc + '/' + youtube_id + '.mp4' + ' -f best ' + url)
    if DL_NAR:
        os.system('youtube-dl -o ' + vid_loc + '/' + youtube_id + ' --write-auto-sub --sub-lang en --convert-subs vtt --skip-download ' + url)