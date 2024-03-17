# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for sitcoms3D dataset.

The dataset is from the paper ["The One Where They Reconstructed 3D Humans and
Environments in TV Shows"](https://ethanweber.me/sitcoms3D/)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Literal

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType, CAMERA_MODEL_TO_TYPE
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class MySemanticDataParserConfig(DataParserConfig):
    """MySemantic dataset parser config"""

    _target: Type = field(default_factory=lambda: MySemantic)
    """target class to instantiate"""
    data: Path = Path("data/MySemantic/my_room_0")
    """Directory specifying location of data."""
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: int = 4
    scene_scale: float = 1.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the MySemantic axis-aligned bbox will be scaled to this value.
    """


@dataclass
class MySemantic(DataParser):
    """MySemantic Dataset"""

    config: MySemanticDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        cameras_json = load_from_json(self.config.data / "transforms.json")
        frames = cameras_json["frames"]

        # bbox = torch.tensor(cameras_json["bbox"])
        aabb_scale = self.config.scene_scale
        bbox = torch.tensor(
            [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
        )

        downscale_suffix = f"_{self.config.downscale_factor}" if self.config.downscale_factor != 1 else ""
        images_folder = f"images{downscale_suffix}"
        segmentations_folder = f"segmentations{downscale_suffix}"

        image_filenames = []
        semantic_filenames = []
        # fx = []
        # fy = []
        # cx = []
        # cy = []
        # height = []
        # width = []
        poses = []
        for frame in frames:
            # unpack data
            image_name = frame["file_path"]  # str
            semantic_name = frame["file_path"][:-3] + 'png'  # str
            image_filename = self.config.data / \
                images_folder / image_name
            semantic_filename = self.config.data / \
                segmentations_folder / semantic_name
            # intrinsics = torch.tensor(frame["intrinsics"])
            poses.append(np.array(frame["transform_matrix"]))
            # append data

            image_filenames.append(image_filename)
            semantic_filenames.append(semantic_filename)
            # fx.append(float(cameras_json["fl_x"]))
            # fy.append(float(cameras_json["fl_y"]))
            # cx.append(float(cameras_json["cx"]))
            # cy.append(float(cameras_json["cy"]))
            # print(float(cameras_json["fl_x"]))
        fx = float(cameras_json["fl_x"])
        fy = float(cameras_json["fl_y"])
        cx = float(cameras_json["cx"])
        cy = float(cameras_json["cy"])
        height = int(cameras_json["h"])
        width = int(cameras_json["w"])

        distortion_params = camera_utils.get_distortion_params(
            k1=float(cameras_json["k1"]) if "k1" in cameras_json else 0.0,
            k2=float(cameras_json["k2"]) if "k2" in cameras_json else 0.0,
            k3=float(cameras_json["k3"]) if "k3" in cameras_json else 0.0,
            k4=float(cameras_json["k4"]) if "k4" in cameras_json else 0.0,
            p1=float(cameras_json["p1"]) if "p1" in cameras_json else 0.0,
            p2=float(cameras_json["p2"]) if "p2" in cameras_json else 0.0,
        )

        # rotate the cameras and box 90 degrees about the x axis to put the z axis up
        # rotation = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        # camera_to_worlds[:, :3] = rotation @ camera_to_worlds[:, :3]
        # bbox = (rotation @ bbox.T).T

        scene_scale = self.config.scene_scale

        # -- set the scene box ---
        scene_box = SceneBox(aabb=bbox)
        # center the box and adjust the cameras too
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # --- semantics ---
        semantics = None
        if self.config.include_semantics:
            empty_path = Path()
            replace_this_path = str(empty_path / images_folder / empty_path)
            with_this_path = str(
                empty_path / segmentations_folder / empty_path)

            filenames = semantic_filenames
            panoptic_classes = load_from_json(
                self.config.data / "semantic_classes.json")
            classes = panoptic_classes["stuff"]
            colors = torch.tensor(
                panoptic_classes["stuff_colors"], dtype=torch.float32) / 255.0
            semantics = Semantics(filenames=filenames,
                                  classes=classes, colors=colors)

        if "camera_model" in cameras_json:
            camera_type = CAMERA_MODEL_TO_TYPE[cameras_json["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )
        cameras.rescale_output_resolution(
            scaling_factor=1.0 / self.config.downscale_factor)

        if "applied_scale" in cameras_json:
            applied_scale = float(cameras_json["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "semantics": semantics} if self.config.include_semantics else {},
            dataparser_scale=scale_factor,
        )
        return dataparser_outputs
