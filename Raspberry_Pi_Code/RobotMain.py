import logging
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from MotorModule import Motor
import KeyboardModule as kp
from CameraModule import Camera

motor = Motor()
cm = Camera()

cm.start()
kp.init()

def main():
    # Capture frame from the camera
    frame = cm.get_frame("BGR")

    # Show the camera feed in a window
    if frame is not None:
        cv2.imshow("Camera Feed", frame)

    # Control motor movements based on keyboard input
    if kp.getKey('UP'):
        motor.move(1, 0, 0.1)
    elif kp.getKey('DOWN'):
        motor.move(-1, 0, 0.1)
    elif kp.getKey('LEFT'):
        motor.move(1, 1, 0.1)
    elif kp.getKey('RIGHT'):
        motor.move(1, -1, 0.1)
    else:
        motor.stop(0.1)

    # Close the camera feed window when a key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        motor.stop(0.1)
        cm.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    while True:
        main()
