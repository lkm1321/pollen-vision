"""Camera configuration class for depthai cameras."""

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import get_socket_from_name, socket_stringToCam

# All ToF *decoding* settings (applied to the depthai v3 ToF node's ToFConfig). Every key here is
# overridable from the "tof_config" block of the camera config json. "median" is a string naming a
# dai.filters.params.MedianFilter member (see utils.median_stringToParam).
TOF_CONFIG_DEFAULTS: Dict[str, Any] = {
    "fps": 30,
    "enable_fppn_correction": True,
    "enable_optical_correction": True,
    "enable_wiggle_correction": True,
    "enable_temperature_correction": False,
    "enable_phase_unwrapping": True,
    "enable_phase_shuffle_temporal_filter": True,
    "enable_burst_mode": False,
    "enable_distortion_correction": True,
    "phase_unwrapping_level": 4,
    "phase_unwrap_error_threshold": 300,
    "median": "KERNEL_3x3",
}

# Post-decode depth filtering (depthai v3 ToFDepthConfidenceFilter + ImageFilters nodes). Overridable
# from the "tof_filtering" block of the camera config json. On RVC2 these run on the host CPU
# (run_on_host), so enabling them costs host cycles rather than device SHAVEs. The image filters are
# applied in the order confidence -> temporal -> speckle -> spatial -> median (Luxonis tuning guide).
TOF_FILTERING_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "run_on_host": True,
    "confidence": {"enable": False, "threshold": 0},
    "temporal": {"enable": False, "alpha": 0.4, "delta": 3, "persistency_mode": "VALID_2_IN_LAST_4"},
    "speckle": {"enable": False, "difference_threshold": 2, "speckle_range": 50},
    "spatial": {"enable": False, "alpha": 0.5, "delta": 3, "hole_filling_radius": 2, "num_iterations": 1},
    "median": "MEDIAN_OFF",
}


def _merge_defaults(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merges a (possibly partial) override dict onto a copy of defaults (one level of nesting)."""
    merged = copy.deepcopy(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


class CamConfig:
    """CamConfig handles some of the camera configuration for depthai cameras.

    Non self explanatory parameters:
    - exposure_params: a tuple of two integers, the first one is the exposure time in microseconds,
                       the second one is the ISO value. Default means auto exposure.
    - isp_scale: a tuple of two integers. The first one is the numerator
                 and the second one is the denominator of the scale factor.
                 This is used to scale the image signal processor (ISP) output.
    - mx_id: the mx_id of the device.
             This allows connecting to multiple devices plugged in the host machine at the same time,
             differentiating them by their mx_id.
    - force_usb2: if True, forces the camera to use USB2 instead of USB3.

    """

    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        resize: Tuple[int, int],
        exposure_params: Optional[Tuple[int, int]],
        mx_id: str = "",
        isp_scale: Tuple[int, int] = (1, 1),
        rectify: bool = False,
        force_usb2: bool = False,
        encoder_quality: int = 80,
        tof_fps: int = 30,
    ) -> None:
        self._cam_config_json = cam_config_json
        self.fps = fps
        self.exposure_params = exposure_params
        if self.exposure_params is not None:
            assert self.exposure_params[0] is not None and self.exposure_params[1] is not None
            iso = self.exposure_params[1]
            assert 100 <= iso <= 1600

        self._mx_id = mx_id
        self.isp_scale = isp_scale
        self.rectify = rectify
        self.force_usb2 = force_usb2
        self.encoder_quality = encoder_quality

        config = json.load(open(self._cam_config_json, "rb"))
        self.socket_to_name = config["socket_to_name"]
        self.inverted = bool(config.get("inverted", False))
        self.fisheye = bool(config.get("fisheye", False))
        self.mono = bool(config.get("mono", False))
        # Config kill switch: "rectify": false forces rectification off regardless of the CLI/constructor.
        # This drops the Warp + NV12-conversion ImageManip nodes from the pipeline (a big cut in
        # RVC2 warp-engine load), so it doubles as a fallback when the full pipeline over-subscribes
        # the device. "fisheye": false selects the plumb_bob (non-fisheye) undistortion model.
        self.rectify = bool(rectify) and bool(config.get("rectify", True))
        # The ToF module lives on its own socket, deliberately kept out of socket_to_name:
        # the stereo code paths (wrapper._prepare(), flash(), ...) assume socket_to_name only
        # contains the identical left/right pair. Its socket is not configured but discovered
        # at runtime (the only sensor reporting CameraSensorType.TOF in supportedTypes).
        self.tof_enabled: bool = bool(config.get("tof", False))
        self.tof_config: Dict[str, Any] = self._parse_tof_config(config, tof_fps)
        self.tof_filtering: Dict[str, Any] = _merge_defaults(TOF_FILTERING_DEFAULTS, config.get("tof_filtering", {}))
        self.tof_socket: Optional[str] = None
        self.tof_fps = int(self.tof_config["fps"])
        self.tof_resolution: Tuple[int, int] = (640, 480)
        self.name_to_socket = {v: k for k, v in self.socket_to_name.items()}
        self.sensor_resolution = (0, 0)
        self.undistort_resolution = (0, 0)
        self.resize_resolution = resize
        self.undistort_maps: Dict[str, Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]] = {
            "left": None,
            "right": None,
        }
        self.calib: dai.CalibrationHandler = dai.CalibrationHandler()

        # lazy init, camera needs to be connected to
        self.P_left: Optional[cv2.UMat] = None
        self.P_right: Optional[cv2.UMat] = None

    def _parse_tof_config(self, config: Dict[str, Any], default_fps: int) -> Dict[str, Any]:
        """Builds the ToF decoding config, layering three sources (lowest to highest precedence):
        TOF_CONFIG_DEFAULTS, the legacy "tof_corrections" bool (kept for backward compatibility), the
        constructor tof_fps, then the explicit "tof_config" block from the camera config json.
        """
        defaults = copy.deepcopy(TOF_CONFIG_DEFAULTS)
        defaults["fps"] = default_fps

        # Legacy shorthand: "tof_corrections": false used to turn off FPPN/wiggle/optical together
        # (fallback for modules whose calibration EEPROM could not be read).
        if "tof_corrections" in config:
            corrections = bool(config["tof_corrections"])
            defaults["enable_fppn_correction"] = corrections
            defaults["enable_wiggle_correction"] = corrections
            defaults["enable_optical_correction"] = corrections

        return _merge_defaults(defaults, config.get("tof_config", {}))

    def get_device_info(self) -> dai.DeviceInfo:
        """Returns a dai.DeviceInfo object with the mx_id.
        This allows connecting to multiple devices plugged in the host machine at the same time,
        differentiating them by their mx_id.
        """

        return dai.DeviceInfo(self._mx_id)

    def set_sensor_resolution(self, resolution: Tuple[int, int]) -> None:
        self.sensor_resolution = resolution

        # Assuming that the resize resolution is the same as the sensor resolution until set otherwise
        if self.resize_resolution is None:
            self.resize_resolution = resolution

    def set_undistort_resolution(self, resolution: Tuple[int, int]) -> None:
        self.undistort_resolution = resolution

    def set_resize_resolution(self, resolution: Tuple[int, int]) -> None:
        self.resize_resolution = resolution

    def set_tof_resolution(self, resolution: Tuple[int, int]) -> None:
        self.tof_resolution = resolution

    def set_tof_socket(self, socket: str) -> None:
        """Records the ToF board socket discovered at runtime (e.g. "CAM_A")."""
        self.tof_socket = socket

    def set_undistort_maps(
        self,
        mapXL: npt.NDArray[np.float32],
        mapYL: npt.NDArray[np.float32],
        mapXR: npt.NDArray[np.float32],
        mapYR: npt.NDArray[np.float32],
    ) -> None:
        self.undistort_maps["left"] = (mapXL, mapYL)
        self.undistort_maps["right"] = (mapXR, mapYR)

    def set_calib(self, calib: dai.CalibrationHandler) -> None:
        self.calib = calib

    def get_calib(self) -> dai.CalibrationHandler:
        """Returns a dai.CalibrationHandler object with all the camera's calibration data."""
        return self.calib

    def get_K_left(self) -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the left camera."""
        left_socket = get_socket_from_name("left", self.name_to_socket)
        left_K = np.array(
            self.calib.getCameraIntrinsics(
                left_socket,
                self.undistort_resolution[0],
                self.undistort_resolution[1],
            )
        )

        return left_K

    def get_K_right(self) -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the right camera."""
        right_socket = get_socket_from_name("right", self.name_to_socket)
        right_K = np.array(
            self.calib.getCameraIntrinsics(
                right_socket,
                self.undistort_resolution[0],
                self.undistort_resolution[1],
            )
        )

        return right_K

    def get_tof_camera_info(self) -> Tuple[int, int, str, List[float], List[float]]:
        """Returns (height, width, distortion_model, D, K) for the ToF camera, for a ROS CameraInfo message.

        Deliberately not reusing to_ROS_msg(), which is stereo-rectification specific
        (it uses the stereo rectification rotation and the P_left/P_right projection matrices).

        If the device EEPROM holds no intrinsics for the ToF socket (Pollen's flash() only writes
        the left/right sockets), approximate intrinsics are synthesized from the module's datasheet
        FoV (90 deg horizontal, 65 deg vertical).
        """
        # Not an assert: the service runs under PYTHONOPTIMIZE=1, which strips asserts, so a None socket
        # would fall through to socket_stringToCam[None] -> KeyError. Fail with a clear message instead.
        if self.tof_socket is None:
            raise RuntimeError(
                "get_tof_camera_info() called but the ToF socket was never discovered. The ToF pipeline "
                "was not built -- set 'tof': true in the camera config json (or drop --tof)."
            )

        width, height = self.tof_resolution
        tof_socket = socket_stringToCam[self.tof_socket]
        distortion_model = "plumb_bob"

        K: Optional[npt.NDArray[np.float64]] = None
        D: List[float] = []
        try:
            K = np.array(self.calib.getCameraIntrinsics(tof_socket, width, height))
            if not np.any(K):
                K = None
            else:
                D = list(self.calib.getDistortionCoefficients(tof_socket))
                if self.calib.getDistortionModel(tof_socket) == dai.CameraModel.Fisheye:
                    distortion_model = "equidistant"
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not read EEPROM intrinsics for ToF socket {self.tof_socket}: {e}")
            K = None

        if K is None:
            logging.getLogger(__name__).warning(
                f"No EEPROM intrinsics for ToF socket {self.tof_socket}, synthesizing approximate intrinsics from the "
                "datasheet FoV (90x65 deg). Flash real ToF intrinsics for accurate reprojection."
            )
            fx = (width / 2) / np.tan(np.deg2rad(90 / 2))
            fy = (height / 2) / np.tan(np.deg2rad(65 / 2))
            K = np.array([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])
            D = [0.0] * 5
            distortion_model = "plumb_bob"

        return height, width, distortion_model, D, list(K.flatten())

    def to_string(self) -> str:
        ret_string = "Camera Config: \n"
        ret_string += "FPS: {}\n".format(self.fps)
        ret_string += "Sensor resolution: {}\n".format(self.sensor_resolution)
        ret_string += "Resize resolution: {}\n".format(self.resize_resolution)
        ret_string += "Inverted: {}\n".format(self.inverted)
        ret_string += "Fisheye: {}\n".format(self.fisheye)
        ret_string += "Mono: {}\n".format(self.mono)
        ret_string += "MX ID: {}\n".format(self._mx_id)
        ret_string += "rectify: {}\n".format(self.rectify)
        ret_string += "force_usb2: {}\n".format(self.force_usb2)
        exp = "auto" if self.exposure_params is None else str(self.exposure_params)
        ret_string += "Exposure params: {}\n".format(exp)
        if self.tof_enabled:
            filt = "filtering on" if self.tof_filtering.get("enabled") else "filtering off"
            tof = "{} @ {} fps, {}".format(self.tof_socket or "socket not discovered yet", self.tof_fps, filt)
        else:
            tof = "none"
        ret_string += "ToF: {}\n".format(tof)
        ret_string += "Undistort maps are: " + "set" if self.undistort_maps["left"] is not None else "not set"

        return ret_string

    def compute_projection_matrices(self) -> Tuple[cv2.UMat, cv2.UMat]:
        left_socket = get_socket_from_name("left", self.name_to_socket)
        right_socket = get_socket_from_name("right", self.name_to_socket)

        left_D = np.array(self.calib.getDistortionCoefficients(left_socket))
        right_D = np.array(self.calib.getDistortionCoefficients(right_socket))

        R = np.array(self.calib.getStereoRightRectificationRotation())

        T = np.array(self.calib.getCameraTranslationVector(left_socket, right_socket))
        T *= 0.01  # to meter for ROS

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.get_K_left(),
            left_D,
            self.get_K_right(),
            right_D,
            self.undistort_resolution,
            R,
            T,
            flags=0,
        )
        return P1, P2

    def to_ROS_msg(
        self, side: str = "left"
    ) -> Tuple[int, int, str, List[float], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        # as defined in https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html

        height = self.resize_resolution[1]
        width = self.resize_resolution[0]
        distortion_model = "plumb_bob"
        if self.calib.getDistortionModel(get_socket_from_name(side, self.name_to_socket)) == dai.CameraModel.Fisheye:
            distortion_model = "equidistant"
        D = self.calib.getDistortionCoefficients(get_socket_from_name(side, self.name_to_socket))

        if self.P_left is None or self.P_right is None:
            self.P_left, self.P_right = self.compute_projection_matrices()

        if side == "left":
            K = self.get_K_left().flatten()
            R = np.array(self.calib.getStereoLeftRectificationRotation()).flatten()
            P = np.array(self.P_left).flatten()

        else:
            K = self.get_K_right().flatten()
            R = np.array(self.calib.getStereoRightRectificationRotation()).flatten()
            P = np.array(self.P_right).flatten()

        return height, width, distortion_model, D, K, R, P
