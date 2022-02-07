import json
import sys
import os
import glob
import tensorflow as tf
import os.path as osp
from absl import logging, flags, app
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import COIN_PATH
from video_encoding.encoding_utils import video_to_frames, image_to_bytes, process_subs

flags.DEFINE_string('mode', 'train', 'define which category of videos to encode train vs. test')

FLAGS = flags.FLAGS

feature = tf.train.Feature
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))

def main(_):
    #%% Initialize dataset in & out directories
    mode = FLAGS.mode
    input_path = osp.join(COIN_PATH, 'videos/%s/'% mode)
    input_path_labels = osp.join(COIN_PATH,'COIN.json')
    
    class_list = sorted(os.listdir(input_path))
    # decode json file to obtain labels
    data = json.load(open(input_path_labels, 'r'))['database']
    # loop through categories in dataset
    count = 0
    for cat in tqdm(class_list):
        if '.' in cat:
            continue
        else:
            
            input_dir = input_path + cat + '/'
            # get all examples per class
            in_dir = glob.glob(input_dir+'*.mp4') #os.listdir(input_dir)
            # create tfrecord save destination
            tfrecord_path = osp.join(COIN_PATH, 'videos_tfrecords/%s_tfrecords' % cat)
            if not osp.exists(tfrecord_path):
                os.makedirs(tfrecord_path)
            record_name = ('%s/%s_%s.tfrecord' %(tfrecord_path,cat, mode))
            print(tfrecord_path)
            print(record_name)
            #input('pause')
            writer = tf.io.TFRecordWriter(record_name)
            #%% create tfrecord per class
            cc = 0
            for filename in in_dir:
                video_filename = osp.basename(filename)
                subtitle_filename = video_filename.replace(".mp4","") + '.en.vtt'
                if (video_filename == '.') or (video_filename == '..'):
                    continue
                #elif cc == 3:
                #    break
                else:
                    #try:
                    cc += 1
                    # decode video
                    video = input_path + cat + '/' + video_filename
                    logging.info('video: %s' %video)
                    # process video
                    frames_list = video_to_frames(video, fps=10, size=224,
                                                  center_crop=True, crop_only=False)
                    ori_fps = 10
                    print(frames_list.shape[0])

                    # reset frame counter and seq_feats for each new video
                    frame_count = 0
                    frames_bytes = []
                    # get frames data
                    for frame in range(frames_list.shape[0]):
                        # save all video frames as bytes
                        img = frames_list[frame,:,:,:]
                        frame_count += 1
                        # convert img to bytes for tfrecord
                        frames_bytes.append(image_to_bytes(img))
                    print(len(frames_list), frame_count)

                    # get video metadata/labels
                    video_name = video_filename.replace(".mp4","")
                    info = data[video_name]
                    start = info['start']
                    end = info['end']
                    duration = info['duration']

                    # get video corresponding step annotations
                    annotation = info['annotation']
                    num_steps = len(annotation)
                    task_labels = []
                    task_ids = []
                    segments_st = []
                    segments_ed = []
                    print(video_filename, num_steps)

                    for i in range(num_steps):
                        anno = annotation[i]
                        task_ids.append(int64_feature([int(anno['id'])]))
                        segments_st.append(float_feature([anno['segment'][0]]))
                        segments_ed.append(float_feature([anno['segment'][1]]))
                        task_labels.append(bytes_feature([str.encode(anno['label'])]))
                    
                    # process subtitles
                    subs_path = input_path + cat + '/' + subtitle_filename
                    if osp.exists(subs_path):
                        caption_texts, caption_starts, caption_ends = process_subs(subs_path)
                    else:
                        caption_texts = [str.encode('no captions')]
                        caption_starts = [-1]
                        caption_ends = [-1]
                    # encode subtitles
                    num_subtitles = len(caption_texts)
                    print(num_subtitles)
                    subtitles = []
                    subtitles_st = []
                    subtitles_ed = []
                    for i in range(num_subtitles):
                        subtitles_st.append(float_feature([caption_starts[i]]))
                        subtitles_ed.append(float_feature([caption_ends[i]]))
                        subtitles.append(bytes_feature([caption_texts[i]]))
                    # add this sequence information into tfrecord
                    seq_feats = {}
                    seq_feats['video'] = tf.train.FeatureList(feature=frames_bytes)
                    seq_feats['steps'] = tf.train.FeatureList(feature=task_labels)
                    seq_feats['steps_ids'] = tf.train.FeatureList(feature=task_ids)
                    seq_feats['steps_st'] = tf.train.FeatureList(feature=segments_st)
                    seq_feats['steps_ed'] = tf.train.FeatureList(feature=segments_ed)
                    seq_feats['subtitles'] = tf.train.FeatureList(feature=subtitles)
                    seq_feats['subtitles_st'] = tf.train.FeatureList(feature=subtitles_st)
                    seq_feats['subtitles_ed'] = tf.train.FeatureList(feature=subtitles_ed)
                    # Create FeatureLists.
                    feature_lists = tf.train.FeatureLists(feature_list=seq_feats)
                    # Add context or video-level features
                    seq_len = frame_count
                    name = str.encode(cat + '_' + video_name)
                    context_features_dict = {'name': bytes_feature([name]),
                                             'len': int64_feature([seq_len]),
                                             'num_steps': int64_feature([num_steps]),
                                             'start': float_feature([start]),
                                             'end': float_feature([end]),
                                             'duration': float_feature([duration]),
                                             'fps': float_feature([ori_fps]),
                                             'num_subtitles': int64_feature([num_subtitles])}
                    context_features = tf.train.Features(feature=context_features_dict)
                    # Create SequenceExample.
                    ex = tf.train.SequenceExample(context=context_features,
                                            feature_lists=feature_lists)
                    writer.write(ex.SerializeToString())
                    #except:
                    #    continue
                    
            writer.close()
            print('total number of videos for category %s is %d' % (cat,cc))
        count = count + cc
        print('total number of encoded videos is %d' % count)

    
if __name__ == '__main__':
    app.run(main)