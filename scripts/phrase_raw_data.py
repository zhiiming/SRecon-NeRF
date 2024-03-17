import os
import shutil
import cv2 as cv
from tqdm import tqdm
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="video processing")

    parser.add_argument("--video-path", default="",
                        help="input path to the video")
    parser.add_argument("--output-folder-path", default="",
                        help="input path to the video")

    args = parser.parse_args()
    return args


def get_image_from_3dscanner_all_data():
    root_path = '/home/ilab/dzm/GitHub/nerfstudio_working_folder/data/myscan'
    raw_path = os.path.join(root_path, 'kb714_low_reflection')
    out_path = os.path.join(root_path, 'images')
    file_names = os.listdir(raw_path)
    for file_name in file_names:
        if file_name.endswith('jpg'):
            src = os.path.join(raw_path, file_name)
            dst = os.path.join(out_path, file_name)
            shutil.copyfile(src, dst)


def extract_frames(video_path, output_folder_path):
    # Open the video file
    video = cv.VideoCapture(video_path)
    # Initialize frame count
    frame_count = 0
    # Read the video frames
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        # Break if no frame is captured
        if not ret:
            break
        # Write the frame as an image file
        output_path = os.path.join(
            output_folder_path, "all_frames", "frame_%04d.jpg" % frame_count)
        cv.imwrite(output_path, frame)
        # Increment frame count
        frame_count += 1
    # Release the video file
    video.release()
    print(f"Frames extracted: {frame_count}")


def select_frames_by_laplacian(output_folder_path, selected_num=300):
    all_frames_path = os.path.join(output_folder_path, "all_frames")
    selected_frames_path = os.path.join(output_folder_path, "images")
    all_frames_name_list = os.listdir(all_frames_path)
    all_frames_name_list.sort()
    selected_frames_name_list = []
    spacing = int(len(all_frames_name_list) / selected_num)
    for idx in tqdm(range(0, len(all_frames_name_list)-spacing-1, spacing)):
        max_fm = -100
        max_frame = -1
        for sub_idx in range(spacing):
            current_frame_name = all_frames_name_list[idx + sub_idx]
            current_frame = cv.imread(os.path.join(
                all_frames_path, current_frame_name))
            gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
            fm = cv.Laplacian(gray, cv.CV_64F)
            fm = fm.var()
            # fm = np.uint(np.absolute(fm))
            if fm > max_fm:
                max_fm = fm
                max_frame = current_frame_name
        assert max_frame != -1
        selected_frames_name_list.append(max_frame)
        src = os.path.join(all_frames_path, max_frame)
        dst = os.path.join(selected_frames_path, max_frame)
        shutil.copyfile(src, dst)
    print(f"{len(selected_frames_name_list)} has been selected.")


def get_image_from_video():
    video_path = '/home/ilab/dzm/GitHub/nerf_for_ipm/data_in/kb714_low_reflection.MOV'
    output_folder_path = '/home/ilab/dzm/GitHub/nerf_for_ipm/data_in/video_kb714_low_reflection'
    if not os.path.exists(output_folder_path):
        print('need to extract frames from video.')
        os.makedirs(output_folder_path)
        os.mkdir(os.path.join(output_folder_path, "all_frames"))
        extract_frames(video_path, output_folder_path)
    if not os.path.exists(os.path.join(output_folder_path, "images")):
        print('need to select frames.')
        os.mkdir(os.path.join(output_folder_path, "images"))
        select_frames_by_laplacian(output_folder_path, selected_num=300)


if __name__ == '__main__':
    args = parse_args()
    # video_path = '/home/ilab/dzm/GitHub/nerf_for_ipm/data_in/kb714_low_reflection.MOV'
    # output_folder_path = '/home/ilab/dzm/GitHub/nerf_for_ipm/data_in/video_kb714_low_reflection'
    video_path = args.video_path
    output_folder_path = args.output_folder_path
    if not os.path.exists(output_folder_path):
        print('need to extract frames from video.')
        os.makedirs(output_folder_path)
        os.mkdir(os.path.join(output_folder_path, "all_frames"))
        extract_frames(video_path, output_folder_path)
    if not os.path.exists(os.path.join(output_folder_path, "images")):
        print('need to select frames.')
        os.mkdir(os.path.join(output_folder_path, "images"))
        select_frames_by_laplacian(output_folder_path, selected_num=50)
