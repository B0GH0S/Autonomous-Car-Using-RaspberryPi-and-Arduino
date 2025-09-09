from CameraModule import Camera
import DataCollectionModule as dcM
import KeyboardModule as kbM
import MotorModule as mM
import cv2
from time import sleep

def steering_test():
    # Initialize modules
    motor = mM.Motor()
    kbM.init()
    
    print("Steering Direction Test")
    print("-----------------------")
    print("Press LEFT arrow to test left steering")
    print("Press RIGHT arrow to test right steering")
    print("Press 'q' to quit")
    
    while True:
        steering = 0
        throttle = 0.1  # Small throttle to see movement
        
        if kbM.getKey("LEFT"):
            steering = -0.5  # Test negative value for left
            print("Testing LEFT steering with value:", steering)
        elif kbM.getKey("RIGHT"):
            steering = 0.5  # Test positive value for right
            print("Testing RIGHT steering with value:", steering)
        
        # Move the motor based on inputs
        motor.move(throttle, steering)
        
        # Exit condition
        if kbM.getKey("q"):
            motor.stop()
            print("Test ended")
            break
        
        cv2.waitKey(1)

if __name__ == "__main__":
    steering_test()