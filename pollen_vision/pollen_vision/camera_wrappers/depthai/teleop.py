from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import (
    imagefilters_preset_stringToMode,
    median_stringToParam,
    socket_camToString,
)
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
        - tof_fps: fallback ToF frame rate when the camera config json does not set tof_config.fps
    """

    # depthai nodes / outputs populated in _create_encoders() and _create_tof_nodes().
    left_encoder: Any
    right_encoder: Any
    left_encoder_mjpeg: Any
    right_encoder_mjpeg: Any
    tof: Any
    _tof_depth_out: Any

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

        self._queues_mjpeg: Dict[str, dai.MessageQueue] = {}

        # The ToF queue deliberately lives outside self.queues: get_data_h264() and the WebRTC
        # path iterate self.queues assuming every entry is an H264 bitstream queue.
        self._queue_tof: Optional[dai.MessageQueue] = None

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

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Overloads the base class abstract method to link the pipeline with the nodes together."""

        self.left_out.link(self.left_encoder.input)
        self.left_out.link(self.left_encoder_mjpeg.input)
        self.right_out.link(self.right_encoder.input)
        self.right_out.link(self.right_encoder_mjpeg.input)

        # The ToF node and its optional filter chain are linked internally in _create_tof_nodes.
        return pipeline

    def _create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates the h264 and mjpeg encoders for the left and right images."""

        profile = dai.VideoEncoderProperties.Profile.H264_BASELINE
        bitrate = 4000
        numBFrames = 0  # no B frames for streaming
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.left_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.left_encoder.setNumBFrames(numBFrames)
        self.left_encoder.setBitrateKbps(bitrate)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.right_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.right_encoder.setNumBFrames(numBFrames)
        self.right_encoder.setBitrateKbps(bitrate)

        profile = dai.VideoEncoderProperties.Profile.MJPEG

        self.left_encoder_mjpeg = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder_mjpeg.setDefaultProfilePreset(self.cam_config.fps, profile)

        self.right_encoder_mjpeg = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder_mjpeg.setDefaultProfilePreset(self.cam_config.fps, profile)

        return pipeline

    def _create_pipeline(self) -> dai.Pipeline:
        """Creates the pipeline for the depthai device.

        Returns the linked pipeline.
        """

        pipeline = self._pipeline_basis()

        pipeline = self._create_encoders(pipeline)

        if self._tof_enabled():
            pipeline = self._create_tof_nodes(pipeline)

        return self._link_pipeline(pipeline)

    def _create_tof_nodes(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates the on-device ToF node and selects which depth output to publish.

        The ToF board socket is not configured but discovered here: it is the only connected sensor
        reporting CameraSensorType.TOF in supportedTypes. (Matching on supportedTypes, not on
        CameraFeatures.name, which is an EEPROM alias and can be empty.)

        The v3 ToF node bundles the depth decoding with a confidence filter and an image-filter chain,
        initialised from an ImageFiltersPresetMode ("tof_filtering.preset") as the starting point.
        "tof_config" (corrections, phase unwrapping, median) applies on-device to the decode. On RVC2 the
        image/confidence filters run on the host CPU, so tof.rawDepth is the decode-only output (no host
        cost) and tof.depth adds the preset filters: "tof_filtering.enabled" selects between them.
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

        tc = self.cam_config.tof_config
        filt = self.cam_config.tof_filtering
        preset = imagefilters_preset_stringToMode.get(
            filt.get("preset", "TOF_MID_RANGE"), dai.ImageFiltersPresetMode.TOF_MID_RANGE
        )
        self.tof = pipeline.create(dai.node.ToF).build(tof_socket, preset, int(tc["fps"]))

        cfg = self.tof.getInitialConfig()
        cfg.enableFPPNCorrection = bool(tc["enable_fppn_correction"])
        cfg.enableOpticalCorrection = bool(tc["enable_optical_correction"])
        cfg.enableWiggleCorrection = bool(tc["enable_wiggle_correction"])
        cfg.enableTemperatureCorrection = bool(tc["enable_temperature_correction"])
        cfg.enablePhaseUnwrapping = bool(tc["enable_phase_unwrapping"])
        cfg.enablePhaseShuffleTemporalFilter = bool(tc["enable_phase_shuffle_temporal_filter"])
        cfg.enableBurstMode = bool(tc["enable_burst_mode"])
        cfg.enableDistortionCorrection = bool(tc["enable_distortion_correction"])
        cfg.phaseUnwrappingLevel = int(tc["phase_unwrapping_level"])
        cfg.phaseUnwrapErrorThreshold = int(tc["phase_unwrap_error_threshold"])
        cfg.setMedianFilter(median_stringToParam[tc["median"]])
        self.tof.setInitialConfig(cfg)

        self._tof_depth_out = self.tof.depth if bool(filt.get("enabled")) else self.tof.rawDepth

        return pipeline

    def _create_queues(self) -> Dict[str, dai.MessageQueue]:
        """Creates the h264 (streaming), mjpeg (ROS) and ToF depth output queues from the node outputs."""

        # config for video: https://docs.luxonis.com/projects/api/en/latest/components/device/#output-queue-maxsize-and-blocking
        queues_h264: Dict[str, dai.MessageQueue] = {}
        queues_h264["left"] = self.left_encoder.bitstream.createOutputQueue(maxSize=30, blocking=True)
        queues_h264["right"] = self.right_encoder.bitstream.createOutputQueue(maxSize=30, blocking=True)

        self._queues_mjpeg["left_mjpeg"] = self.left_encoder_mjpeg.bitstream.createOutputQueue(maxSize=1, blocking=False)
        self._queues_mjpeg["right_mjpeg"] = self.right_encoder_mjpeg.bitstream.createOutputQueue(maxSize=1, blocking=False)

        if self._tof_enabled():
            self._queue_tof = self._tof_depth_out.createOutputQueue(maxSize=1, blocking=False)

        return queues_h264
