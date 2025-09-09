from CameraModule import Camera
import DataCollectionModule as dcM
import KeyboardModule as kbM
import MotorModule as mM
import cv2
from time import sleep

# Initialize modules
maxThrottle = 0.25
motor = mM.Motor()
kbM.init()
wM = Camera()
wM.start()

record = 0
while True:
    steering = 0
    throttle = 0
    
    if kbM.getKey("LEFT"):
        steering = -0.8 
    elif kbM.getKey("RIGHT"):
        steering = 0.8
    
    if kbM.getKey("UP"):
        throttle = maxThrottle  # Forward
    elif kbM.getKey("DOWN"):
        throttle = -maxThrottle  # Backward
    
    # Recording controls
    if kbM.getKey("r"):  # Start/stop recording with 'R' key
        if record == 0:
            print('Recording Started ...')
        record += 1
        sleep(0.300)
    
    if record == 1:
        img = wM.get_frame()
        dcM.saveData(img, -steering)
    elif record == 2:
        dcM.saveLog()
        record = 0

    # Move the motor based on inputs
    motor.move(throttle, -steering)
    
    # Exit condition (press 'q' to quit)
    if kbM.getKey("q"):
        motor.stop()
        break
    
    cv2.waitKey(1)