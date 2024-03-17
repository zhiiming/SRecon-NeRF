#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--dataset_dir", default="",
                        help="input path to the dataset")
    parser.add_argument("--image_prefix", default="frame",
                        help="prefix of image file")
    parser.add_argument("--semantic_prefix", default="frame",
                        help="prefix of segmentation label")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # args.dataset_dir = '/home/ilab/dzm/GitHub/nerf_for_ipm/data_in/video_kb714'

    in_dataset_dir = args.dataset_dir
    image_prefix = args.image_prefix
    semantic_prefix = args.semantic_prefix
    scale_factors = [2, 4, 8]
    dataset_name = in_dataset_dir.split('/')[-1]
    root_path = os.path.dirname(os.path.dirname(in_dataset_dir))
    out_dataset_dir = os.path.join(root_path, 'dataset', dataset_name)

    if os.path.exists(out_dataset_dir):
        shutil.rmtree(out_dataset_dir)
    os.mkdir(out_dataset_dir)

    # copy transforms.json
    src_transforms_path = os.path.join(in_dataset_dir, 'transforms_all.json')
    dst_transforms_path = os.path.join(out_dataset_dir, 'transforms.json')
    shutil.copyfile(src_transforms_path, dst_transforms_path)

    # copy semantic_classes.json
    src_json_path = os.path.join(in_dataset_dir, 'semantic_classes.json')
    dst_json_path = os.path.join(out_dataset_dir, 'semantic_classes.json')
    shutil.copyfile(src_json_path, dst_json_path)

    # write images / segmantations
    src_image_path = os.path.join(in_dataset_dir, 'images')
    src_semantic_path = os.path.join(in_dataset_dir, 'segmentations')
    dst_image_path = os.path.join(out_dataset_dir, 'images')
    dst_semantic_path = os.path.join(out_dataset_dir, 'segmentations')
    os.mkdir(dst_image_path)
    os.mkdir(dst_semantic_path)
    semantic_names = os.listdir(src_semantic_path)

    new_frames = []
    json_file = json.load(open(src_transforms_path))
    for frame in json_file["frames"]:
        frame_name = frame["file_path"]
        seg_name = frame_name[:-3] + 'png'
        if seg_name not in semantic_names:
            continue
        shutil.copyfile(os.path.join(src_image_path, frame_name),
                        os.path.join(dst_image_path, frame_name))
        shutil.copyfile(os.path.join(src_semantic_path, seg_name),
                        os.path.join(dst_semantic_path, seg_name))
        new_frames.append(frame)
    json_file["frames"] = new_frames
    # shutil.copytree(src_image_path, dst_image_path)
    # shutil.copytree(src_semantic_path, dst_semantic_path)
    print(f"[INFO] writing {len(new_frames)} frames to {dst_transforms_path}")
    with open(dst_transforms_path, "w") as outfile:
        json.dump(json_file, outfile, indent=2)

    # write the downscaled images / segmantations
    image_file_names = os.listdir(os.path.join(out_dataset_dir, 'images'))
    for scale_factor in scale_factors:
        out_image_folder = os.path.join(
            out_dataset_dir, 'images_' + str(scale_factor))
        out_segmentation_folder = os.path.join(
            out_dataset_dir, 'segmentations_' + str(scale_factor))
        os.mkdir(out_image_folder)
        os.mkdir(out_segmentation_folder)

        for idx, image_name in enumerate(image_file_names):
            # image_path = os.path.join(in_dataset_dir, 'images', '%s_%04d.jpg' % (image_prefix, idx))
            image_path = os.path.join(in_dataset_dir, 'images', image_name)
            semantic_path = os.path.join(
                in_dataset_dir, 'segmentations', image_name[:-3] + 'png')
            # semantic_path = os.path.join(
            #     in_dataset_dir, 'segmentations', '%s_%04d.png' % (semantic_prefix, idx))
            image = Image.open(image_path)
            semantic = Image.open(semantic_path)
            width, height = image.size
            newsize = (int(width / scale_factor), int(height / scale_factor))
            image = image.resize(newsize, resample=Image.NEAREST)
            semantic = semantic.resize(newsize, resample=Image.NEAREST)
            image_out_path = os.path.join(out_image_folder, image_name)
            semantic_out_path = os.path.join(
                out_segmentation_folder, image_name[:-3] + 'png')
            image.save(image_out_path)
            semantic.save(semantic_out_path)
