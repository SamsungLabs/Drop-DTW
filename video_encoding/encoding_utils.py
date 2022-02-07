import io
import csv
import webvtt
import ffmpeg
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
import numpy as np


# prepare variables/ functions for tfrecord creation
feature = tf.train.Feature
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))


def video_to_frames(video_filename, fps=10, size=224, center_crop=True, crop_only=False, no_crop=False):
    cmd = (
        ffmpeg
        .input(video_filename)
        .filter('fps', fps=fps)
    )
    if no_crop:
        probe = ffmpeg.probe(video_filename)
        video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video['width'])
        height = int(video['height'])
    else:
        width, height = size, size
        if center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(size, aw),
                            '(ih - {})*{}'.format(size, ah),
                            str(size), str(size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                            '(ih - min(iw,ih))*{}'.format(ah),
                            'min(iw,ih)',
                            'min(iw,ih)')
                .filter('scale', size, size)
            )
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    # produces and array of size [T, W, H, 3]
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return frames


# functions to handle gt annotations
def Time2FrameNumber(t, ori_fps, fps=10):
    """ function to convert segment annotations given in seconds to frame numbers
        input:
            ori_fps: is the original fps of the video
            fps: is the fps that we are using to extract frames from the video
            num_frames: is the number of frames in the video (under fps)
            t: is the time (in seconds) that we want to convert to frame number 
        output: 
            numf: the frame number corresponding to the time t of a video encoded at fps
    """
    ori2fps_ratio = int(ori_fps/fps)
    ori_numf = t*ori_fps
    numf = int(ori_numf / ori2fps_ratio)
    return numf


def time2seconds(timestamp):
    """ convert timestamps in the format xx:xx:xx.xx to seconds"""
    h,m,s = timestamp.split(':')
    timestamp_sec = float(h)*3600 + float(m)*60 + float(s)
    return timestamp_sec


def process_subs(sub_path):
    caption_texts = []
    caption_starts = []
    caption_ends = []
    for caption in webvtt.read(sub_path):
        caption_dur = float(caption.end[-5:]) - float(caption.start[-5:])
        if caption_dur > 0.5:
            caption_text = caption.text
            caption_text = caption_text.replace('\n', ' ')
            caption_text = caption_text.strip(' ')
            caption_texts.append(str.encode(caption_text))
            caption_start = time2seconds(caption.start)
            caption_starts.append(caption_start)
            caption_end = time2seconds(caption.end)
            caption_ends.append(caption_end)
    return caption_texts, caption_starts, caption_ends


def read_task_files(task_file):
    " read task files and save into dictionary"
    file = open(task_file, 'r')
    Lines = file.readlines()
    # initialize dictionary
    tasks_dict = dict()
    count = 0
    for line in Lines:
        if count > 5:
            count = 0
        line = line.strip()
        #print(count, line)
        
        if count == 0:
            task_id = line
            tasks_dict[task_id] = dict()
        elif count == 1:
            tasks_dict[task_id]['task_name'] = line
        elif count == 3:
            num_steps = int(line)
            tasks_dict[task_id]['num_steps'] =  num_steps
        elif count == 4:
            annot = line.split(',')
            annotations = []
            for j in range(num_steps):
                annotations.append(annot[j])
            tasks_dict[task_id]['annotations'] = annotations
        
        count +=1
    return tasks_dict


def read_anno_files(anno_path):
    """ read annotations csv files with 3 entries 
    [step_id, step_st, step_ed]"""
    anno = [] 
    with open(anno_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            anno.append(row)
    return anno



def image_to_bytes(image_array):
    """Get bytes formatted image arrays."""
    image = Image.fromarray(image_array)
    im_string = bytes_feature([image_to_jpegstring(image)])
    return im_string


def image_to_jpegstring(image, jpeg_quality=95):
    """Convert image to a JPEG string."""
    if not isinstance(image, Image.Image):
        raise TypeError('Provided image is not a PIL Image object')
    # This fix to PIL makes sure that we don't get an error when saving large
    # jpeg files. This is a workaround for a bug in PIL. The value should be
    # substantially larger than the size of the image being saved.
    ImageFile.MAXBLOCK = 640 * 512 * 64
    output_jpeg = io.BytesIO()
    image.save(output_jpeg, 'jpeg', quality=jpeg_quality, optimize=True)
    return output_jpeg.getvalue()


def get_video_length(input_video):
    import subprocess
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)