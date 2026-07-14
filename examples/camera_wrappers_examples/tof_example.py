"""Standalone ToF acceptance test: opens the teleop head and shows a live colorized depth window.

The config json must contain "tof": true (the ToF board socket is auto-detected at runtime).
Depth values are uint16 millimeters. Press 'q' to exit.
"""

import argparse
import logging

import cv2
import numpy as np
from pollen_vision.camera_wrappers.depthai.teleop import TeleopWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="ToF example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()

logging.basicConfig(level=logging.INFO)

w = TeleopWrapper(get_config_file_path(args.config), 60, rectify=False)

while True:
    depth, lat, _ = w.get_data_tof()
    logging.info(f"latency: {lat}, center depth: {depth[depth.shape[0] // 2, depth.shape[1] // 2]} mm")

    non_zero = depth[depth != 0]
    min_depth = np.percentile(non_zero, 1) if non_zero.size > 0 else 0
    max_depth = np.percentile(non_zero, 99) if non_zero.size > 0 else 1
    depth_normalized = np.interp(depth, (min_depth, max_depth), (0, 255)).astype(np.uint8)
    depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    cv2.imshow("tof depth (mm)", depth_colorized)
    if cv2.waitKey(1) == ord("q"):
        break
