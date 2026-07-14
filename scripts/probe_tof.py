"""Hardware discovery script for the head ToF module (Phase 0 of the ToF integration).

Run this on the robot with the streaming service stopped (a depthai device is exclusive
to one process). For every connected device it prints, per camera socket:
- the board socket
- the sensor name and resolution
- whether the sensor supports ToF (this is how the wrapper auto-detects the ToF socket
  at runtime when the camera config json has "tof": true)
- whether the device EEPROM holds intrinsics for that socket
"""

import logging

import depthai as dai
import numpy as np
from pollen_vision.camera_wrappers.depthai.utils import socket_camToString

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for device_info in dai.Device.getAllAvailableDevices():
    device = dai.Device(device_info)
    logger.info(f"Device {device_info.getMxId()}")
    calib = device.readCalibration()

    for cam in device.getConnectedCameraFeatures():
        socket = socket_camToString[cam.socket]
        is_tof = dai.CameraSensorType.TOF in cam.supportedTypes
        logger.info(
            f"  socket={socket} sensor={cam.sensorName} resolution={cam.width}x{cam.height} "
            f"alias='{cam.name}' supportedTypes={cam.supportedTypes} tof={is_tof}"
        )

        try:
            K = np.array(calib.getCameraIntrinsics(cam.socket, cam.width, cam.height))
            if np.any(K):
                logger.info(f"  EEPROM intrinsics found:\n{K}")
            else:
                logger.warning(f"  EEPROM intrinsics for {socket} are all zero")
        except Exception as e:
            logger.warning(f"  No EEPROM intrinsics for {socket}: {e}")

        if is_tof:
            logger.info(f'  >>> ToF found on socket {socket}. Set "tof": true in the camera config json to enable it <<<')

    device.close()
