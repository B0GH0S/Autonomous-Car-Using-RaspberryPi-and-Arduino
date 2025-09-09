import cv2
import numpy as np
from tensorflow.keras.models import load_model

import CameraModule as wM
import MotorModule as mM
import KeyboardModule as kM

steeringSen = 0.57
maxThrottle = 0.5
cm = wM.Camera()
cm.start()
kM.init()
motor = mM.Motor()
model = load_model('/home/pi/Desktop/car/src/model.keras')

def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

while True:
    img = cm.get_frame()
    img = np.asarray(img)
    img = preProcess(img)
    img_array = np.array([img])
    steering = float(model.predict(img_array, verbose=0))
    steering_adj = steering * steeringSen
        
    print(steering_adj)
    motor.move(maxThrottle, steering_adj)
    if kM.getKey("q"):
        break
    cv2.waitKey(1)
    
cm.stop()
motor.stop()
cv2.destroyAllWindows()