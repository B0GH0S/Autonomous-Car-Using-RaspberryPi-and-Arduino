import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import albumentations as A
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import random

#### STEP 1 - INITIALIZE DATA
def getName(filePath):
    return os.path.join(*filePath.split('/')[-2:])

def importDataInfo(path):
    columns = ['Center', 'Steering']
    data = pd.DataFrame()
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            try:
                dataNew = pd.read_csv(os.path.join(path, file), names=columns)
                print(f'{file}: {dataNew.shape[0]}')
                
                # Validate image paths
                dataNew['Center'] = dataNew['Center'].apply(
                    lambda x: os.path.join(path, x) if os.path.exists(os.path.join(path, x)) else None
                )
                dataNew = dataNew.dropna(subset=['Center'])
                data = pd.concat([data, dataNew], ignore_index=True)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print('Total Images Imported:', data.shape[0])
    return data

#### STEP 2 - VISUALIZE AND BALANCE DATA
def balanceData(data, display=True):
    nBin = 31
    samplesPerBin = 700
    hist, bins = np.histogram(data['Steering'], nBin)
    
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    
    removeindexList = []
    for j in range(nBin):
        binDataList = [i for i in range(len(data['Steering'])) 
                      if bins[j] <= data['Steering'][i] <= bins[j+1]]
        binDataList = shuffle(binDataList)[samplesPerBin:]
        removeindexList.extend(binDataList)
    
    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    
    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    return data

#### STEP 3 - PREPARE FOR PROCESSING
def loadData(path, data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        img_path = os.path.join(path, indexed_data[0])
        if os.path.exists(img_path):
            imagesPath.append(img_path)
            steering.append(float(indexed_data[1]))
        else:
            print(f"Warning: Missing image {img_path}")
    return np.asarray(imagesPath), np.asarray(steering)

#### STEP 5 - AUGMENT DATA
def augmentImage(imgPath, steering):
    try:
        img = cv2.imread(imgPath)
        if img is None:
            raise FileNotFoundError(f"Could not read image {imgPath}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
        ])
        augmented = transform(image=img)
        img = augmented["image"]
        
        if 'horizontal_flip' in augmented and augmented["horizontal_flip"]:
            steering = -steering
            
        return img, steering
    except Exception as e:
        print(f"Error augmenting {imgPath}: {e}")
        return None, None

#### STEP 6 - PREPROCESS
def preProcess(img):
    img = img[54:120, :, :]  # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img / 255.0  # Normalize

#### STEP 7 - CREATE MODEL (UPDATED)
def createModel():
    model = Sequential([
        Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'),
        Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Convolution2D(64, (3, 3), activation='elu'),
        Convolution2D(64, (3, 3), activation='elu'),
        Flatten(),
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1)
    ])
    model.compile(Adam(learning_rate=0.0001), loss=MeanSquaredError())
    return model

#### STEP 8 - TRAINING
def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        
        for _ in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = cv2.imread(imagesPath[index])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering = steeringList[index]
            
            if img is not None:
                img = preProcess(img)
                imgBatch.append(img)
                steeringBatch.append(steering)
        
        if len(imgBatch) > 0:  # Ensure we don't yield empty batches
            yield (np.asarray(imgBatch), np.asarray(steeringBatch))
