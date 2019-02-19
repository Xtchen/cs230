"""This module has util functions that help pre-process training data.
1. convert RGB to gray scale. 2. detect and crop mouth area using dlib."""
from glob import glob
from multiprocessing import Pool
from shutil import copyfile
from time import time
import os

import cv2
import dlib
import numpy as np
import skvideo.io

"""The directory of the LRS3 data set."""
SOURCE_DIR = '/home/xiaotongchen/workspace/stanford/cs230/data/lrs3/pretrain'
"""The directory where you want to output the cropped data set."""
DEST_DIR = '/home/xiaotongchen/workspace/stanford/cs230/data/lrs3/cropped'
"""25 Hz"""
FPS = 25
"""Resolution is 120 * 120"""
FRAME_ROWS = 120
FRAME_COLS = 120
"""Covert RGB channels to gray scale."""
COLORS = 1
"""To speed up training, we should skip long videos. (500 / 25 = 20 sec)"""
MAX_FRAMES_COUNT = 500


def chunkify(lst, n):
    """Cut the data set into n small ones evenly."""
    return [lst[i::n] for i in range(n)]


def crop_all():
    """Process data using multiple processes."""
    source_dirs = sorted(os.listdir(SOURCE_DIR))
    chuncks = chunkify(source_dirs, 8)
    print(f'in total {len(chuncks)} chuncks: {len(i) for i in chuncks}')

    with Pool(8) as p:
        print(p.map(crop_dirs, chuncks))


def crop_dirs(source_dirs):
    """https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat"""
    predictor_path = '~/dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for idx, source_dir in enumerate(source_dirs):
        print(f'processing {idx} / {len(source_dirs)}')
        dest_dir = os.path.join(DEST_DIR, source_dir)

        if not os.path.exists(dest_dir):
            os.mkdir(os.path.join(DEST_DIR, source_dir))

        all_mp4 = sorted(glob(os.path.join(SOURCE_DIR, source_dir, '*.mp4')))
        all_txt = sorted(glob(os.path.join(SOURCE_DIR, source_dir, '*.txt')))

        time_start = time()

        for mp4 in all_mp4:
            crop_video(mp4, detector, predictor)
        #
        # for txt in all_txt:
        #     process_txt(txt)

        time_end = time()

        print(f'takes {time_end - time_start}')


# def process_txt(txt_path):
#     parts = txt_path.split(os.sep)
#     dir_name, txt_name = parts[-2:]  # e.g. DwYQHj7Hmik, 00026.txt
#     copyfile(txt_path, os.path.join(DEST_DIR, dir_name, txt_name))


def crop_video(input_video_path, detector, predictor):
    """See https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/"""
    parts = input_video_path.split(os.sep)
    dir_name, video_name = parts[-2:]  # e.g. DwYQHj7Hmik, 00026.mp4
    dest_path = os.path.join(DEST_DIR, dir_name, video_name)

    if os.path.exists(dest_path):
        print(f'file: {dest_path} exists. skipping')
        return

    cap = cv2.VideoCapture(input_video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    if FPS != video_fps:
        print("video FPS is not 25 Hz")
        return

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_counter = MAX_FRAMES_COUNT
    num_frames = min(total_num_frames, max_counter)

    if total_num_frames > max_counter:
        return

    counter = 0
    valid_num_frames = 0

    # Required parameters for mouth extraction.
    width_crop_max = 0
    height_crop_max = 0

    temp_frames = np.zeros((num_frames, FRAME_ROWS, FRAME_COLS), dtype="uint8")

    for i in np.arange(num_frames):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == 0:
            break
        if counter > num_frames:
            break

        # detect using frontal_face_detector
        detections = detector(frame, 1)

        # 20 mark for mouth: features 48 - 68
        marks = np.zeros((2, 20))

        if len(detections) > 0:
            for k, d in enumerate(detections):

                # Shape of the face.
                shape = predictor(frame, d)

                co = 0
                # Specific for the mouth.
                for ii in range(48, 68):
                    feature = shape.part(ii)
                    marks[0, co] = feature.x
                    marks[1, co] = feature.y
                    co += 1

                # calculate the left-most and right-most points.
                x_left = int(np.amin(marks, axis=1)[0])
                y_left = int(np.amin(marks, axis=1)[1])
                x_right = int(np.amax(marks, axis=1)[0])
                y_right = int(np.amax(marks, axis=1)[1])

                # calculate the center of the mouth.
                x_center = (x_left + x_right) / 2
                y_center = (y_left + y_right) / 2
                width = x_right - x_left + 60
                height = y_right - y_left + 60
                org_width = x_right - x_left
                org_height = y_right - y_left

                if width_crop_max == 0 and height_crop_max == 0:
                    width_crop_max = width
                    height_crop_max = height
                else:
                    width_crop_max += 1.5 * np.maximum(org_width - width_crop_max, 0)
                    height_crop_max += 1.5 * np.maximum(org_height - height_crop_max, 0)

                x_left_crop = int(x_center - width_crop_max / 2)
                x_right_crop = int(x_center + width_crop_max / 2)
                y_left_crop = int(y_center - height_crop_max / 2)
                y_right_crop = int(y_center + height_crop_max / 2)

                if x_left_crop >= 0 and y_left_crop >= 0 and x_right_crop < w and y_right_crop < h:
                    mouth = frame[y_left_crop:y_right_crop, x_left_crop:x_right_crop, :]
                    mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                    mouth_gray = cv2.resize(mouth_gray, (FRAME_COLS, FRAME_ROWS))
                    temp_frames[i * COLORS:i * COLORS + COLORS, :, :] = mouth_gray
                    valid_num_frames += 1

        counter += 1

    temp_frames = temp_frames.astype(np.uint8)
    keep_ratio = valid_num_frames / num_frames
    print(f'temp_frames shape: {temp_frames.shape} ratio: {keep_ratio}')

    print(f'writing file: {dest_path}')
    skvideo.io.vwrite(dest_path, temp_frames)
    copyfile(input_video_path.replace('mp4', 'txt'), os.path.join(DEST_DIR, dir_name, video_name.replace('mp4', 'txt')))
    with open(os.path.join(DEST_DIR, dir_name, video_name.replace('mp4', 'ratio')), 'w') as f:
        f.write(str(keep_ratio))


if __name__ == "__main__":
    crop_all()
