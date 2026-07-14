from datetime import timedelta
from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import socket_camToString
from pollen_vision.camera_wrappers.depthai.wrapper import DepthaiWrapper


class TeleopWrapper(DepthaiWrapper):  # type: ignore[misc]
    """A wrapper for the depthai library that exposes only the relevant features for Pollen's teleoperation feature.

    Calling get_data() returns h264 encoded left and right images.

    Args:
        - cam_config_json: path to the camera configuration json file
        - fps: frames per second
        - force_usb2: force the use of USB2
        - rectify: rectify the images using the calibration data stored in the eeprom of the camera
        - exposure_params: tuple of two integers (exposure, gain) to set the exposure and gain of the camera
        - mx_id: the id of the camera
    """

    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool = False,
        rectify: bool = False,
        exposure_params: Optional[Tuple[int, int]] = None,
        mx_id: str = "",
        tof_fps: int = 30,
    ) -> None:
        self._data_h264: Dict[str, npt.NDArray[np.uint8]] = {}
        self._latency_h264: Dict[str, float] = {}
        self._ts_h264: Dict[str, timedelta] = {}

        self._data_mjpeg: Dict[str, npt.NDArray[np.uint8]] = {}
        self._latency_mjpeg: Dict[str, float] = {}
        self._ts_mjpeg: Dict[str, timedelta] = {}

        self._queues_mjpeg: Dict[str, dai.DataOutputQueue] = {}

        # The ToF queue deliberately lives outside self.queues: get_data_h264() and the WebRTC
        # path iterate self.queues assuming every entry is an H264 bitstream queue.
        self._queue_tof: Optional[dai.DataOutputQueue] = None

        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=(960, 720),
            rectify=rectify,
            exposure_params=exposure_params,
            mx_id=mx_id,
            isp_scale=(2, 3),
            tof_fps=tof_fps,
        )

    def get_data_h264(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        """Extends the get_data method of the Wrapper class to return the h264 encoded left and right images.

        Returns:
            - Tuple(data, latency, timestamp) : Tuple of dictionaries of h264 encoded left and right images,
                latencies and timestamps for each camera.
        """

        for name, queue in self.queues.items():
            pkt = queue.get()
            self._data_h264[name] = pkt.getData()
            self._latency_h264[name] = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[call-arg]
            self._ts_h264[name] = pkt.getTimestamp()

        return self._data_h264, self._latency_h264, self._ts_h264

    def get_data_mjpeg(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        for name, queue in self._queues_mjpeg.items():
            pkt = queue.get()
            self._data_mjpeg[name] = pkt.getData()  # type: ignore[attr-defined]
            self._latency_mjpeg[name] = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[attr-defined, call-arg]
            self._ts_mjpeg[name] = pkt.getTimestamp()  # type: ignore[attr-defined]

        return self._data_mjpeg, self._latency_mjpeg, self._ts_mjpeg

    def get_data_tof(self) -> Tuple[npt.NDArray[np.uint16], timedelta, timedelta]:
        """Returns the latest ToF depth frame (2D uint16, depth in millimeters), its latency and its timestamp.

        Blocking call, paced by the ToF fps: meant to run in its own host thread, like get_data_mjpeg().
        """
        if self._queue_tof is None:
            raise RuntimeError("ToF is not enabled (no 'tof': true in the camera config json)")

        pkt = self._queue_tof.get()
        frame: npt.NDArray[np.uint16] = pkt.getFrame()  # type: ignore[attr-defined]
        latency: timedelta = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[attr-defined, call-arg]

        return frame, latency, pkt.getTimestamp()  # type: ignore[attr-defined]

    def _tof_enabled(self) -> bool:
        return bool(self.cam_config.tof_enabled)

    def _create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        super()._create_output_streams(pipeline)

        self.xout_left_mjpeg = pipeline.createXLinkOut()
        self.xout_left_mjpeg.setStreamName("left_mjpeg")

        self.xout_right_mjpeg = pipeline.createXLinkOut()
        self.xout_right_mjpeg.setStreamName("right_mjpeg")

        if self._tof_enabled():
            self.xout_tof = pipeline.createXLinkOut()
            self.xout_tof.setStreamName("tof_depth")

        return pipeline

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Overloads the base class abstract method to link the pipeline with the nodes together."""

        self.left.isp.link(self.left_manip.inputImage)
        self.left_manip.out.link(self.left_encoder.input)
        self.left_manip.out.link(self.left_encoder_mjpeg.input)
        self.left_encoder.bitstream.link(self.xout_left.input)
        self.right_encoder.bitstream.link(self.xout_right.input)

        self.right.isp.link(self.right_manip.inputImage)
        self.right_manip.out.link(self.right_encoder.input)
        self.right_manip.out.link(self.right_encoder_mjpeg.input)
        self.left_encoder_mjpeg.bitstream.link(self.xout_left_mjpeg.input)
        self.right_encoder_mjpeg.bitstream.link(self.xout_right_mjpeg.input)

        if self._tof_enabled():
            self.tof_cam.raw.link(self.tof.input)
            self.tof.depth.link(self.xout_tof.input)

        return pipeline

    def _create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates the h264 encoders for the left and right images."""

        profile = dai.VideoEncoderProperties.Profile.H264_BASELINE
        bitrate = 4000
        numBFrames = 0  # no B frames for streaming
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.left_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.left_encoder.setNumBFrames(numBFrames)
        self.left_encoder.setBitrateKbps(bitrate)
        # self.left_encoder.setQuality(self.cam_config.encoder_quality)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.right_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.right_encoder.setNumBFrames(numBFrames)
        self.right_encoder.setBitrateKbps(bitrate)
        # self.right_encoder.setQuality(self.cam_config.encoder_quality)

        profile = dai.VideoEncoderProperties.Profile.MJPEG

        self.left_encoder_mjpeg = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder_mjpeg.setDefaultProfilePreset(self.cam_config.fps, profile)
        # self.left_encoder_mjpeg.setLossless(True)

        self.right_encoder_mjpeg = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder_mjpeg.setDefaultProfilePreset(self.cam_config.fps, profile)
        # self.right_encoder_mjpeg.setLossless(True)

        return pipeline

    def _create_pipeline(self) -> dai.Pipeline:
        """Creates the pipeline for the depthai device.

        Returns the linked pipeline.
        """

        pipeline = self._pipeline_basis()

        pipeline = self._create_encoders(pipeline)

        if self._tof_enabled():
            pipeline = self._create_tof_nodes(pipeline)

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_tof_nodes(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates the ToF camera node and the on-device ToF depth decoding node.

        The ToF board socket is not configured but discovered here: it is the only connected sensor
        reporting CameraSensorType.TOF in supportedTypes. (Matching on supportedTypes, not on
        CameraFeatures.name, which is an EEPROM alias and can be empty.)
        The raw ToF stream is decoded on-device (dai.node.ToF) into a uint16 depth map in millimeters.
        """
        tof_socket: Optional[dai.CameraBoardSocket] = None
        for cam in self._device.getConnectedCameraFeatures():
            if dai.CameraSensorType.TOF in cam.supportedTypes:
                tof_socket = cam.socket
                self.cam_config.set_tof_socket(socket_camToString[cam.socket])
                self.cam_config.set_tof_resolution((cam.width, cam.height))
                self._logger.info(f"ToF sensor '{cam.sensorName}' found on socket {self.cam_config.tof_socket}")
                break

        if tof_socket is None:
            raise RuntimeError(
                "'tof': true is set in the camera config json but no connected camera reports a ToF sensor "
                "(run scripts/probe_tof.py to inspect the device)."
            )

        self.tof_cam = pipeline.create(dai.node.Camera)
        self.tof_cam.setBoardSocket(tof_socket)
        self.tof_cam.setFps(self.cam_config.tof_fps)

        self.tof = pipeline.create(dai.node.ToF)
        tof_config = self.tof.initialConfig.get()
        corrections = bool(self.cam_config.tof_corrections)
        if not corrections:
            self._logger.warning(
                "ToF EEPROM-based corrections (FPPN/wiggle/optical) disabled by config: depth accuracy will be degraded"
            )
        tof_config.enableFPPNCorrection = corrections
        tof_config.enableOpticalCorrection = corrections
        tof_config.enableWiggleCorrection = corrections
        tof_config.enablePhaseShuffleTemporalFilter = True
        tof_config.enablePhaseUnwrapping = True
        tof_config.phaseUnwrappingLevel = 4
        tof_config.phaseUnwrapErrorThreshold = 300
        tof_config.enableTemperatureCorrection = False  # not stable at depthai 2.27
        tof_config.median = dai.MedianFilter.KERNEL_3x3
        self.tof.initialConfig.set(tof_config)

        return pipeline

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        """Extends the base class method _create_queues() to add the h264 encoded left and right images queues."""

        # config for video: https://docs.luxonis.com/projects/api/en/latest/components/device/#output-queue-maxsize-and-blocking
        queues_h264: Dict[str, dai.DataOutputQueue] = {}
        for name in ["left", "right"]:
            queues_h264[name] = self._device.getOutputQueue(name, maxSize=30, blocking=True)

        for name in ["left_mjpeg", "right_mjpeg"]:
            self._queues_mjpeg[name] = self._device.getOutputQueue(name, maxSize=1, blocking=False)

        if self._tof_enabled():
            self._queue_tof = self._device.getOutputQueue("tof_depth", maxSize=1, blocking=False)

        return queues_h264
