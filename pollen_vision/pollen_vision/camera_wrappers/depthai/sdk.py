from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import get_socket_from_name
from pollen_vision.camera_wrappers.depthai.wrapper import DepthaiWrapper


# Depth is left aligned by convention
# TODO do we need to give the option to change this?
# NOTE: migrated to the depthai v3 API alongside the teleop path. The stereo-depth branch
# (compute_depth=True) has NOT been validated on a device yet — it is not used by the Reachy teleop
# head. Validate on an SR / depth-capable camera before relying on it.
class SDKWrapper(DepthaiWrapper):  # type: ignore[misc]
    """A wrapper for the depthai library that exposes only the relevant features for Pollen's reachy sdk.

    Calling get_data() returns the left and right images, and if compute_depth is True:
    - returns the depth and disparity maps
    - returns the rectified left and right images from the depthai's depth node (grayscale)

    If jpeg_output is True, the left and right images are encoded in mjpeg.

    Args:
        - cam_config_json: path to the camera configuration json file
        - fps: frames per second
        - force_usb2: force the use of USB2
        - resize: tuple of two integers (width, height) to resize the images
        - rectify: rectify the images using the calibration data stored in the eeprom of the camera
        - compute_depth: compute the depth and disparity maps
        - exposure_params: tuple of two integers (exposure, gain) to set the exposure and gain of the camera
        - mx_id: the id of the camera
        - jpeg_output: encode the left and right images in mjpeg
    """

    # depthai nodes populated in _create_encoders() / _create_pipeline().
    left_encoder: Any
    right_encoder: Any
    depth: Any
    depth_max_disparity: Any

    def __init__(
        self,
        cam_config_json: str,
        fps: int = 30,
        force_usb2: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        rectify: bool = False,  # TODO Not working when compute_depth is True for now
        compute_depth: bool = False,
        exposure_params: Optional[Tuple[int, int]] = None,
        mx_id: str = "",
        jpeg_output: bool = False,
        encoder_quality: int = 95,
    ) -> None:
        self._compute_depth = compute_depth
        self._mjpeg = jpeg_output
        assert not (self._compute_depth and rectify), "Rectify is not working when compute_depth is True for now"

        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=resize if not compute_depth else (1280, 800),
            rectify=rectify if not compute_depth else False,
            exposure_params=exposure_params,
            mx_id=mx_id,
            encoder_quality=encoder_quality,
        )

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        """Extends the base class method get_data() to return opencv frames.
        Returns:
            - Tuple(data, latency, timestamp) : Tuple of dictionaries of opencv frames,
                latencies and timestamps for each camera.
        """
        data, latency, ts = super().get_data()
        for name, pkt in data.items():
            data[name] = pkt.getCvFrame()

        return data, latency, ts

    def get_K(self) -> npt.NDArray[np.float32]:
        return super().get_K(left=True)  # type: ignore

    def get_depth_K(self) -> npt.NDArray[np.float32]:
        return super().get_K(left=True)  # type: ignore

    def _create_queues(self) -> Dict[str, dai.MessageQueue]:
        """Extends the base class method _create_queues() to add the depth and disparity queues
        as well as the rectified left and right images queues from depthai's depth node.
        """

        queues: Dict[str, dai.MessageQueue] = super()._create_queues()
        if self._compute_depth:
            queues["depth"] = self.depth.depth.createOutputQueue(maxSize=1, blocking=False)
            queues["disparity"] = self.depth.disparity.createOutputQueue(maxSize=1, blocking=False)

            queues["depthNode_left"] = self.depth.rectifiedLeft.createOutputQueue(maxSize=1, blocking=False)
            queues["depthNode_right"] = self.depth.rectifiedRight.createOutputQueue(maxSize=1, blocking=False)

        return queues

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Overloads the base class abstract method _link_pipeline() to link the nodes together.

        Sets self._out_left / self._out_right (consumed by the base _create_queues) to either the
        mjpeg-encoded bitstream or the raw camera/warp output.
        """

        if self._mjpeg:
            self.left_out.link(self.left_encoder.input)
            self.right_out.link(self.right_encoder.input)
            self._out_left = self.left_encoder.bitstream
            self._out_right = self.right_encoder.bitstream
        else:
            self._out_left = self.left_out
            self._out_right = self.right_out

        if self._compute_depth:
            self.left_out.link(self.depth.left)
            self.right_out.link(self.depth.right)

        return pipeline

    def _create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates the mjpeg encoders."""
        profile = dai.VideoEncoderProperties.Profile.MJPEG
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.left_encoder.setQuality(self.cam_config.encoder_quality)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.right_encoder.setQuality(self.cam_config.encoder_quality)

        return pipeline

    def _create_pipeline(self) -> dai.Pipeline:
        """Overloads the base class abstract method _create_pipeline() to create the pipeline.
        Sets up the basic pipeline with the left and right cameras from the base class method _pipeline_basis()
        and adds the depth node if compute_depth is True, and the mjpeg encoders if mjpeg is True.
        Returns the linked pipeline.
        """

        pipeline = self._pipeline_basis()
        self.left.initialControl.setSharpness(0)
        self.left.initialControl.setLumaDenoise(0)
        self.left.initialControl.setChromaDenoise(0)

        self.right.initialControl.setSharpness(0)
        self.right.initialControl.setLumaDenoise(0)
        self.right.initialControl.setChromaDenoise(0)

        if self._compute_depth:
            # Configuring depth node (depthai v3: StereoDepthConfig is mutated in place, there is no
            # initialConfig.get()/set(), and the v2 HIGH_DENSITY preset is DENSITY in v3).
            left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
            self.depth = pipeline.create(dai.node.StereoDepth)
            self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
            self.depth.setLeftRightCheck(True)
            self.depth.setExtendedDisparity(False)
            self.depth.setSubpixel(True)
            self.depth.setDepthAlign(left_socket)
            self.depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            self.depth_max_disparity = self.depth.initialConfig.getMaxDisparity()

            post = self.depth.initialConfig.postProcessing
            post.speckleFilter.enable = False
            post.speckleFilter.speckleRange = 50
            post.temporalFilter.enable = False
            post.spatialFilter.enable = False
            post.spatialFilter.holeFillingRadius = 2
            post.spatialFilter.numIterations = 1

        if self._mjpeg:
            pipeline = self._create_encoders(pipeline)

        return self._link_pipeline(pipeline)
