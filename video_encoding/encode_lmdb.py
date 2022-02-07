import sys
import os
import torch
import pickle
import pandas
import json
import argparse
import numpy as np
import tensorflow as tf

import lmdb
import pyarrow as pa

from os import path as osp
from tqdm import tqdm
from glob import glob

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from video_encoding.models.encoder import Encoder
from paths import COIN_PATH


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, help="root folder of tfrecords")
parser.add_argument('-d', '--dest', type=str, help="root folder for storing lmdb folder/files")
parser.add_argument('--dataset', type=str, default='COIN', help="name of the dataset we are encoding")
parser.add_argument('--num_parts', type=int, default=1, help="in how many parts to split encoding")
parser.add_argument('--part', type=int, default=1, help="the current part of encoding. The value \in [1, num_parts]")


device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Encoder()
encoder.to(device)


def decode(serialized_example):
    """Decode serialized SequenceExample."""

    context_features = {
      'name': tf.io.FixedLenFeature([], dtype=tf.string),
      'len': tf.io.FixedLenFeature([], dtype=tf.int64),
      'num_steps': tf.io.FixedLenFeature([], dtype=tf.int64),
      'duration': tf.io.FixedLenFeature([], dtype=tf.float32),
      'fps': tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    seq_features = {}

    seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    seq_features['steps'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    seq_features['steps_ids'] = tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
    seq_features['steps_st'] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    seq_features['steps_ed'] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
  
    # Extract features from serialized data.
    context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=seq_features)
  
    name = tf.cast(context_data['name'], tf.string)
    seq_len = context_data['len']
    num_steps = context_data['num_steps']
    duration = context_data['duration']
    fps = context_data['fps']

    video = sequence_data.get('video', [])
    steps = sequence_data.get('steps', [])
    steps_ids = sequence_data.get('steps_ids', [])
    steps_st = sequence_data.get('steps_st', [])
    steps_ed = sequence_data.get('steps_ed', [])

    return seq_len, name,  num_steps, duration, fps,\
                     video, steps, steps_ids, steps_st, steps_ed


def sample_and_preprocess(seq_len, name, num_steps, duration, fps,
                          video, steps, steps_ids, steps_st, steps_ed,):
    """Samples frames and prepares them for training."""
    # Decode the encoded JPEG images
    
    video = tf.map_fn( tf.image.decode_jpeg, video, dtype=tf.uint8)

    return {'frames': video, 'steps': steps, 'steps_ids': steps_ids, 'steps_st': steps_st,
            'steps_ed': steps_ed, 'seq_len': seq_len, 'name': name, 'num_steps': num_steps,
            'duration': duration, 'fps': fps}


def process_tfrec_dict(video_dict, dataset):
    vd = video_dict
    
    # general video-level info
    name = vd['name'].numpy().decode("utf-8") 
    if dataset == 'COIN':
        activity_name, activity_id = name.split('_')[:2]
    elif dataset == 'CrossTask':
        activity_name = activity_id = name.split('_')[0]
    elif dataset == 'YouCook2':
        activity_name = activity_id = name.split('_')[0]
    
    cls_id = int(activity_id)
    
    # video frames embeddings
    frames = vd['frames'].numpy()
    frames = frames / 255
    video_frames = torch.from_numpy(frames).to(torch.float32).to(device)
    frames_features = encoder.embed_full_video(video_frames).cpu().numpy()

    # create embeddings for the text that corresponds to steps. Something like:
    step_texts = [s.decode("utf-8") for s in vd['steps'].numpy()]
    step_features = encoder.embed_full_subs(step_texts).cpu().numpy()

    # subtitles ebmeddings
    try:
        subs_texts = [s.decode("utf-8") for s in vd['subtitles'].numpy()]
        subs_features = encoder.embed_full_subs(subs_texts).cpu().numpy()
        subs_starts = vd['subtitles_st'].numpy()
        subs_ends = vd['subtitles_ed'].numpy()
        num_subs = vd['num_subtitles'].numpy()
    except:
        # no encoded subs found
        subs_features = np.zeros([1, 512])
        subs_starts = -np.ones([1])
        subs_ends = -np.ones([1])
        num_subs = np.zeros([0])

    # steps info
    num_steps = vd['num_steps'].numpy()
    steps_ids = vd['steps_ids'].numpy()
    steps_starts = vd['steps_st'].numpy()
    steps_ends = vd['steps_ed'].numpy()
    
    sample = {
        'name': name,
        'cls': cls_id,
        'cls_name': activity_name,
        'frames_features': frames_features,
        'steps_features': step_features,
        'subs_features': subs_features,
        'subs_starts': subs_starts,
        'subs_ends': subs_ends,
        'num_subs': num_subs,
        'num_steps': num_steps,
        'steps_ids': steps_ids,
        'steps_starts': steps_starts,
        'steps_ends': steps_ends
    }
    return sample


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def encode_lmdb(tf_dataset, lmdb_path, dataset, write_frequency=500):
    isdir = osp.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    
    txn = db.begin(write=True)
    names = []
    for idx, tfrec_sample in enumerate(tf_dataset.take(-1)):
        video_sample = process_tfrec_dict(tfrec_sample, dataset=dataset)
        name = video_sample['name']
        names.append(name)
        txn.put(u'{}'.format(name).encode('ascii'), dumps_pyarrow(video_sample))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in names]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def encode_folder(source, dest, dataset_name):
    tfrecords = glob(osp.join(source, '*.tfrecord'))
    folder_name = osp.basename(source.rstrip('/'))
    lmdb_dest = osp.join(dest, folder_name).replace('tfrecords', 'lmdb')
    if not osp.isdir(lmdb_dest):
        os.mkdir(lmdb_dest)

    # assuming lmdb would be stored in the same parent folder as of tfrecords
    for tfrecord in tfrecords:
        tfrecord_name = osp.basename(tfrecord)
        lmdb_path = osp.join(lmdb_dest, tfrecord_name.replace('tfrecord', 'lmdb'))

        dataset = tf.data.TFRecordDataset(tfrecord)
        dataset = dataset.map(decode)
        dataset = dataset.map(sample_and_preprocess)

        encode_lmdb(tf_dataset=dataset, lmdb_path=lmdb_path, dataset=dataset_name)


def encode_step_info(dataset):
    if dataset == 'COIN':
        data_path = COIN_PATH 
        # read_annotations
        with open(osp.join(data_path, 'COIN.json')) as f:
            annots = json.load(f)
        annots = annots['database']
        taxonomy_steps = pandas.read_csv(osp.join(data_path, 'taxonomy_step.csv'))
    else:
        assert "This dataset is not supported"

    cls_to_type = dict()
    cls_to_steps = dict()
    steps_to_cls = dict()
    steps_to_embeddings = dict()
    steps_to_descriptions = dict()
    for row in taxonomy_steps.iterrows():
        cls_id = row[1]['Target Id']
        step_id = row[1]['Action Id']
        step_description = row[1]['Action Label']
        if cls_id not in cls_to_steps:
            cls_to_steps[cls_id] = []
        cls_to_steps[cls_id].append(step_id)
        steps_to_cls[step_id] = cls_id
        steps_to_descriptions[step_id] = step_description
        cls_to_type[cls_id] = 'primary'
        
        with torch.no_grad():
            step_embedding = encoder.net.text_module([step_description])['text_embedding'].cpu().numpy()
        steps_to_embeddings[step_id] = step_embedding
        
    steps_info = {
                  'cls_to_steps': cls_to_steps,
                  'step_to_cls': steps_to_cls,
                  'steps_to_embeddings': steps_to_embeddings,
                  'steps_to_descriptions': steps_to_descriptions,
                  'cls_to_type': cls_to_type,
                }

    filename = os.path.join(data_path, 'steps_info.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(steps_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parser.parse_args()
    class_folders = glob(osp.join(args.source, '*/'))
    part_size = int(np.ceil(len(class_folders) / args.num_parts))
    part_start, part_end = part_size * (args.part - 1), part_size * args.part
    folders_to_encode = class_folders[part_start:part_end]
    print('\n Folders to encode:', folders_to_encode, '\n')
    for tfrec_folder in tqdm(folders_to_encode):
        print(f"Encoding: {tfrec_folder}")
        encode_folder(tfrec_folder, args.dest, args.dataset)
    print('\n Encoding steps info', '\n')
    encode_step_info(args.dataset)
    print('Done')
