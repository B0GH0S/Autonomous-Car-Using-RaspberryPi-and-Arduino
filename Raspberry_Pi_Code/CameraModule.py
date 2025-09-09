import logging
import time
import cv2
import numpy as np
from picamera2 import Picamera2


class CameraError(Exception):
    """Custom exception class for camera errors."""
    pass


class Camera:
    def __init__(self, resolution=(480, 240)):
        """
        Initialize the Camera class with the specified resolution.
        :param resolution: Tuple specifying the resolution (width, height) of the camera feed.
        """
        if not isinstance(resolution, tuple) or len(resolution) != 2 or not all(isinstance(i, int) for i in resolution):
            raise ValueError("Resolution must be a tuple of two integers (width, height).")

        self.picam2 = Picamera2()
        self.resolution = resolution
        self.running = False

        # Configure the camera
        preview_config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": self.resolution},
            buffer_count=4,
        )
        self.picam2.configure(preview_config)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Camera initialized with resolution {self.resolution}.")

    def start(self):
        """
        Start the camera.
        """
        if self.running:
            self.logger.warning("Camera is already running.")
            return

        try:
            self.picam2.start()
            self.running = True
            time.sleep(2)
            self.logger.info("Camera started successfully.")
        except Exception as e:
            raise CameraError(f"Failed to start the camera: {e}")

    def stop(self):
        """
        Stop the camera and release resources.
        """
        if not self.running:
            self.logger.warning("Camera is not running.")
            return

        try:
            self.picam2.stop()
            self.running = False
            self.logger.info("Camera stopped.")
        except Exception as e:
            raise CameraError(f"Failed to stop the camera: {e}")

    def get_frame(self, format="BGR"):
        """
        Capture a frame from the camera.
        :param format: The desired frame format ("BGR" or "GRAY").
        :return: The captured frame in the specified format.
        """
        if not self.running:
            raise CameraError("Camera is not running. Call start() before capturing frames.")
        try:
            frame = self.picam2.capture_array()
            if format.upper() == "GRAY":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif format.upper() == "BGR":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError("Unsupported format. Use 'BGR' or 'GRAY'.")
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None

    def __del__(self):
        """
        Destructor to ensure the camera is stopped.
        """
        if self.running:
            self.logger.info("Destructor called. Stopping the camera.")
            self.stop()
