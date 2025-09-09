import pandas as pd
import os
import cv2
from datetime import datetime
from CameraModule import Camera  # Adjust the import according to the module name

# Initialize the global variables
global imgList, steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

# GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'DataCollected')

# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory, f'IMG{str(countFolder)}')):
    countFolder += 1
newPath = myDirectory + "/IMG" + str(countFolder)
os.makedirs(newPath)

# SAVE IMAGES IN THE FOLDER
def saveData(img, steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    fileName = os.path.join(newPath, f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    steeringList.append(steering)

# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog():
    global imgList, steeringList
    rawData = {'Image': imgList,
               'Steering': steeringList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory, f'log_{str(countFolder)}.csv'), index=False, header=False)
    print('Log Saved')
    print('Total Images: ', len(imgList))

# Main function using Camera class
def main():
    # Initialize Camera class with the desired resolution
    camera = Camera(resolution=(640, 480))
    camera.start()

    for x in range(10):  # Capture 10 images as demo
        img = camera.get_frame(format="BGR")
        if img is not None:
            steering = 0.5  # Example steering value (adjust as needed)
            saveData(img, steering)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

    saveLog()
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()