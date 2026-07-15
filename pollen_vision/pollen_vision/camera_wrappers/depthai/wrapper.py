"""Depthai Wrapper module.
"""

import json
import os
import sys
from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from pollen_vision.camera_wrappers.depthai.calibration.undistort import compute_undistort_maps, get_mesh
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import get_inv_R_T, get_socket_from_name, socket_camToString


class DepthaiWrapper(CameraWrapper):  # type: ignore
    """Wrapper is an abstract class for luxonis cameras using the depthai library.

    It factors out the common code between the different camera wrappers.

    Migrated to the depthai v3 API: the pipeline owns the device, cameras are created with
    Camera.build()/requestOutput(), rectification uses the dedicated Warp node, and output queues
    are created directly from node outputs (there is no XLinkOut / stream-name indirection).
    """

    # depthai graph handles, populated in _prepare() / _pipeline_basis() and the subclass hooks.
    # left_out / right_out are the (rectified) camera outputs; _out_left / _out_right are the
    # terminal outputs the base _create_queues() reads (set by the subclass in _link_pipeline()).
    pipeline: dai.Pipeline
    left: Any
    right: Any
    left_out: Any
    right_out: Any
    _out_left: Any
    _out_right: Any

    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool,
        resize: Tuple[int, int],
        rectify: bool,
        exposure_params: Optional[Tuple[int, int]],
        mx_id: str,
        isp_scale: Tuple[int, int] = (1, 1),
        encoder_quality: int = 80,
        tof_fps: int = 30,
    ) -> None:
        super().__init__()

        # --- DYNAMIC VR OVERRIDE ---
        custom_resize = resize
        custom_quality = encoder_quality

        # Inspect the raw boot command for the VR client argument
        if "UnityClient" in sys.argv:
            custom_resize = (640, 480)
            custom_quality = 50
        # ---------------------------

        self.cam_config = CamConfig(
            cam_config_json,
            fps,
            custom_resize,
            exposure_params,
            mx_id,
            isp_scale,
            rectify,
            force_usb2,
            encoder_quality=custom_quality,
            tof_fps=tof_fps,
        )

        self._prepare()

    def _prepare(self) -> None:
        """Prepares the camera for use.

        Sets up :
        - camera configuration
        - device connection
        - pipeline (nodes added to a v3 pipeline that owns the device)
        - output queues

        If requested, pre-computes the undistort maps for the rectification.

        """
        self._logger.debug("Connecting to camera")

        self._device = dai.Device(
            self.cam_config.get_device_info(),
            maxUsbSpeed=(dai.UsbSpeed.HIGH if self.cam_config.force_usb2 else dai.UsbSpeed.SUPER_PLUS),
        )

        connected_cameras_features = []
        for cam in self._device.getConnectedCameraFeatures():
            if socket_camToString[cam.socket] in self.cam_config.socket_to_name.keys():
                connected_cameras_features.append(cam)

        # Assuming both cameras are the same
        width = connected_cameras_features[0].width
        height = connected_cameras_features[0].height

        self.cam_config.set_sensor_resolution((width, height))

        # Note: doing this makes the teleopWrapper not work with cams other than the teleoperation head.
        # This comes from the (2, 3) ispscale factor that is not appropriate for 1280x800 resolution.
        # Not really a big deal
        width_undistort_resolution = int(width * (self.cam_config.isp_scale[0] / self.cam_config.isp_scale[1]))
        height_unistort_resolution = int(height * (self.cam_config.isp_scale[0] / self.cam_config.isp_scale[1]))
        self.cam_config.set_undistort_resolution((width_undistort_resolution, height_unistort_resolution))
        self.cam_config.set_calib(self._device.readCalibration())

        if self.cam_config.rectify:
            self._set_undistort_maps()

        # depthai v3: the pipeline is constructed around the already-opened device, and the subclass
        # populates it with nodes in _create_pipeline().
        self.pipeline = dai.Pipeline(self._device)
        # Restored from the v2 pipeline (still valid in v3): chunk size 0 sends each XLink packet in a
        # single transfer, needed for the throughput of the encoded + raw ToF streams over USB.
        self.pipeline.setXLinkChunkSize(0)
        self._create_pipeline()

        # Output queues are created from node outputs and must exist before the pipeline is started.
        self.queues = self._create_queues()

        try:
            self.pipeline.start()
        except Exception as e:
            if self.cam_config.tof_enabled:
                raise RuntimeError(
                    "Could not start the depthai pipeline with the ToF enabled. This may be an RVC2 resource "
                    "exhaustion (the ToF decoding and host-run filters share resources with the video encoders). "
                    "Try lowering tof_config.fps, disabling tof_filtering, or removing 'tof': true from the camera "
                    "config json to disable the ToF."
                ) from e
            raise

        self.print_info()

    def print_info(self) -> None:
        """Prints the camera configuration."""
        self._logger.info(self.cam_config.to_string())

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        """Gets the data from the camera.

        Returns a tuple containing the data, the latency and the timestamp.
        data is a dict containing the left and right images as well as the depth map if it exists.
        The content of data is defined by the queues created in the _create_queues method.
        """

        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        for name, queue in self.queues.items():
            pkt = queue.get()
            data[name] = pkt  # type: ignore[assignment]
            latency[name] = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[attr-defined, call-arg]
            ts[name] = pkt.getTimestamp()  # type: ignore[attr-defined]

        return data, latency, ts

    def get_K(self, left: bool = True) -> npt.NDArray[np.float32]:
        return self.cam_config.get_K_left() if left else self.cam_config.get_K_right()  # type: ignore[no-any-return]

    @abstractmethod
    def _create_pipeline(self) -> dai.Pipeline:
        """Abstract method that is implemented by the subclasses."""

        self._logger.error("Abstract class DepthaiWrapper does not implement create_pipeline()")
        exit()

    def _pipeline_basis(self) -> dai.Pipeline:
        """Creates and configures the left and right cameras (and, if rectifying, their warp nodes).

        Sets self.left / self.right (the Camera nodes) and self.left_out / self.right_out (the
        Node.Output to consume for each side, already rectified when rectify is enabled).
        This method is used (and/or extended) by the subclasses to create the basis pipeline.
        """

        self._logger.debug("Configuring depthai pipeline")
        pipeline = self.pipeline

        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        self.left = pipeline.create(dai.node.Camera).build(left_socket)
        self.right = pipeline.create(dai.node.Camera).build(right_socket)

        for cam in (self.left, self.right):
            if self.cam_config.exposure_params is not None:
                cam.initialControl.setManualExposure(*self.cam_config.exposure_params)
            if self.cam_config.inverted:
                cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        # v3 Camera.requestOutput folds the old ColorCamera ISP-scale + ImageManip resize into one
        # call, delivering a frame already scaled to the undistort resolution.
        width, height = self.cam_config.undistort_resolution
        fps = float(self.cam_config.fps)

        if self.cam_config.rectify:
            # The Warp node does not accept NV12 (its supported inputs are RAW8/GRAY8/RAW16/RGB-BGR
            # planar/YUV420p), so request YUV420p, warp it, then convert to NV12 for the encoders
            # inside _create_warp() -- the v2 ImageManip warp did the warp and the NV12 conversion
            # in a single node.
            left_src = self.left.requestOutput((width, height), dai.ImgFrame.Type.YUV420p, fps=fps)
            right_src = self.right.requestOutput((width, height), dai.ImgFrame.Type.YUV420p, fps=fps)
            self.left_out = self._create_warp("left", left_src, (width, height))
            self.right_out = self._create_warp("right", right_src, (width, height))
        else:
            self.left_out = self.left.requestOutput((width, height), dai.ImgFrame.Type.NV12, fps=fps)
            self.right_out = self.right.requestOutput((width, height), dai.ImgFrame.Type.NV12, fps=fps)

        return pipeline

    @abstractmethod
    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Abstract method that is implemented by the subclasses.
        Links the nodes together.
        """

        self._logger.error("Abstract class DepthaiWrapper does not implement link_pipeline()")
        exit()

    def _create_queues(self) -> Dict[str, dai.MessageQueue]:
        """Creates the output queues from the terminal left/right outputs set by the subclass.
        This method is used (and/or extended) by the subclasses.
        """
        queues: Dict[str, dai.MessageQueue] = {}
        queues["left"] = self._out_left.createOutputQueue(maxSize=1, blocking=False)
        queues["right"] = self._out_right.createOutputQueue(maxSize=1, blocking=False)
        return queues

    def _create_warp(
        self,
        cam_name: str,
        src_out: dai.Node.Output,
        resolution: Tuple[int, int],
    ) -> dai.Node.Output:
        """Rectifies src_out (a YUV420p output) with the precomputed warp mesh and converts the
        warped frame to NV12 for the video encoders (replaces the v2 ImageManip warp path, which did
        both the mesh warp and the NV12 conversion in a single node)."""

        warp = self.pipeline.create(dai.node.Warp)
        try:
            mesh, mesh_width, mesh_height = get_mesh(self.cam_config, cam_name)
            warp.setWarpMesh(mesh, mesh_width, mesh_height)
        except Exception as e:
            self._logger.error(e)
            exit()
        warp.setOutputSize(resolution[0], resolution[1])
        warp.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)
        src_out.link(warp.inputImage)

        # Warp preserves the input frame type (YUV420p); the encoders need NV12, so convert here.
        to_nv12 = self.pipeline.create(dai.node.ImageManip)
        to_nv12.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        to_nv12.initialConfig.setOutputSize(resolution[0], resolution[1], dai.ImageManipConfig.ResizeMode.NONE)
        to_nv12.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)
        warp.out.link(to_nv12.inputImage)
        return to_nv12.out

    def _set_undistort_maps(self) -> None:
        """Computes and assign the undistort maps for the rectification."""
        mapXL, mapYL, mapXR, mapYR = compute_undistort_maps(self.cam_config)
        self.cam_config.set_undistort_maps(mapXL, mapYL, mapXR, mapYR)

    # Takes in the output of multical calibration
    def flash(self, calib_json_file: str) -> None:
        """Flashes the calibration to the camera.

        The calibration is read from the calib_json_file and flashed into the camera's eeprom.
        """
        now = str(datetime.now()).replace(" ", "_").split(".")[0]

        device_calibration_backup_file = Path("./CALIBRATION_BACKUP_" + now + ".json")
        deviceCalib = self._device.readCalibration()
        deviceCalib.eepromToJsonFile(device_calibration_backup_file)
        self._logger.info(f"Backup of device calibration saved to {device_calibration_backup_file}")

        os.environ["DEPTHAI_ALLOW_FACTORY_FLASHING"] = "235539980"

        ch = dai.CalibrationHandler()
        calibration_data = json.load(open(calib_json_file, "rb"))

        cameras = calibration_data["cameras"]
        camera_poses = calibration_data["camera_poses"]

        self._logger.info("Setting intrinsics ...")
        for cam_name, params in cameras.items():
            K = np.array(params["K"])
            D = np.array(params["dist"]).reshape((-1))
            im_size = params["image_size"]
            cam_socket = get_socket_from_name(cam_name, self.cam_config.name_to_socket)

            ch.setCameraIntrinsics(cam_socket, K.tolist(), im_size)
            ch.setDistortionCoefficients(cam_socket, D.tolist())
            if self.cam_config.fisheye:
                self._logger.info("Setting camera type to fisheye ...")
                ch.setCameraType(cam_socket, dai.CameraModel.Fisheye)

        self._logger.info("Setting extrinsics ...")
        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        right_to_left = camera_poses["right_to_left"]
        R_right_to_left = np.array(right_to_left["R"])
        T_right_to_left = np.array(right_to_left["T"])
        T_right_to_left *= 100  # Needs to be in centimeters (?) # TODO test

        R_left_to_right, T_left_to_right = get_inv_R_T(R_right_to_left, T_right_to_left)

        ch.setCameraExtrinsics(
            left_socket,
            right_socket,
            R_right_to_left.tolist(),
            T_right_to_left.tolist(),
            specTranslation=T_right_to_left.tolist(),
        )
        ch.setCameraExtrinsics(
            right_socket,
            left_socket,
            R_left_to_right,
            T_left_to_right,
            specTranslation=T_left_to_right,
        )

        ch.setStereoLeft(left_socket, np.eye(3).tolist())
        ch.setStereoRight(right_socket, R_right_to_left.tolist())

        self._logger.info("Flashing ...")
        try:
            # depthai v3 dropped flashCalibration2(); flashCalibration() is the current API.
            self._device.flashCalibration(ch)
            self._logger.info("Calibration flashed successfully")
        except Exception as e:
            self._logger.error("Flashing failed")
            self._logger.error(e)
            exit()


if __name__ == "__main__":
    from pollen_vision.camera_wrappers.depthai import SDKWrapper
    from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path, get_connected_devices

    devices = get_connected_devices()
    print(f"Detected cameras: {devices}")
    for mxid, name in devices.items():
        if name == "other":
            cam = SDKWrapper(
                get_config_file_path("CONFIG_SR"),
                compute_depth=True,
                rectify=False,
                mx_id=mxid,
                jpeg_output=True,
            )
        else:
            cam = SDKWrapper(
                get_config_file_path("CONFIG_IMX296"),
                compute_depth=False,
                rectify=True,
                mx_id=mxid,
                jpeg_output=True,
            )
