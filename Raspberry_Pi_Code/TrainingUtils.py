import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random

# =============================================
# STEP 1 - DATA LOADING (OPTIMIZED FOR RASPBERRY PI)
# =============================================
def getName(filePath):
    """Extracts relative path from full path"""
    return os.path.join(*filePath.split('/')[-2:])

def importDataInfo(path, max_samples=None):
    """
    Loads CSV data with validation checks.
    Args:
        path: Directory containing CSV files
        max_samples: Optional cap for debugging
    Returns:
        DataFrame with valid image paths and steering angles
    """
    columns = ['Center', 'Steering']
    data = pd.DataFrame()
    
    for file in sorted(os.listdir(path)):
        if file.endswith('.csv'):
            try:
                file_path = os.path.join(path, file)
                data_new = pd.read_csv(file_path, names=columns)
                
                # Validate paths
                data_new['Center'] = data_new['Center'].apply(
                    lambda x: os.path.join(path, x) if os.path.exists(os.path.join(path, x)) else None
                )
                data_new = data_new.dropna(subset=['Center'])
                
                if max_samples:
                    data_new = data_new.sample(min(max_samples, len(data_new)))
                
                data = pd.concat([data, data_new], ignore_index=True)
                print(f"Loaded {file}: {len(data_new)} samples")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nTotal valid samples: {len(data)}")
    return data

# =============================================
# STEP 2 - DATA BALANCING (ENHANCED VISUALIZATION)
# =============================================
def balanceData(data, display=True, samples_per_bin=300, n_bins=31):
    """
    Balances steering angles to prevent bias.
    Args:
        data: Pandas DataFrame
        display: Show before/after plots
        samples_per_bin: Max samples per steering bin
        n_bins: Number of histogram bins
    Returns:
        Balanced DataFrame
    """
    hist, bins = np.histogram(data['Steering'], n_bins)
    
    if display:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((bins[0], bins[-1]), (samples_per_bin, samples_per_bin), 'r-')
        plt.title('Original Data Distribution')
        plt.xlabel('Steering Angle')
        plt.ylabel('Count')
    
    # Remove excess samples
    remove_indices = []
    for j in range(n_bins):
        bin_indices = np.where(
            (data['Steering'] >= bins[j]) & 
            (data['Steering'] <= bins[j+1])
        )[0]
        bin_indices = shuffle(bin_indices)[samples_per_bin:]
        remove_indices.extend(bin_indices)
    
    balanced_data = data.drop(data.index[remove_indices])
    
    if display:
        plt.subplot(1,2,2)
        hist, _ = np.histogram(balanced_data['Steering'], bins)
        plt.bar(center, hist, width=0.03)
        plt.plot((bins[0], bins[-1]), (samples_per_bin, samples_per_bin), 'r-')
        plt.title('Balanced Data Distribution')
        plt.xlabel('Steering Angle')
        plt.suptitle(f"Removed {len(remove_indices)} samples | Final: {len(balanced_data)}")
        plt.tight_layout()
        plt.show()
    
    return balanced_data

# =============================================
# STEP 3 - DATA PREPROCESSING (OPTIMIZED FOR PI)
# =============================================
def loadData(path, data):
    """Validates and loads image paths with steering angles"""
    images, steerings = [], []
    for _, row in data.iterrows():
        img_path = os.path.join(path, row['Center'])
        if os.path.exists(img_path):
            images.append(img_path)
            steerings.append(float(row['Steering']))
        else:
            print(f"Missing image: {img_path}")
    return np.array(images), np.array(steerings)

# =============================================
# STEP 4 - AUGMENTATION (REALISTIC DRIVING SCENARIOS)
# =============================================
def augmentImage(imgPath, steering):
    """
    Applies realistic driving augmentations using Albumentations
    Returns:
        Augmented image and adjusted steering angle
    """
    try:
        img = cv2.imread(imgPath)
        if img is None:
            raise FileNotFoundError(f"Could not read {imgPath}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Enhanced augmentation pipeline
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=5,  # Small realistic rotations
                p=0.7
            ),
            A.RandomShadow(p=0.3),
            A.RandomRain(p=0.1),
            A.GaussNoise(var_limit=(10,50)),])
        
        
        augmented = transform(image=img)
        img = augmented["image"]
        
        # Adjust steering if flipped
        if 'horizontal_flip' in augmented and augmented["horizontal_flip"]:
            steering = -steering
            
        return img, steering
    except Exception as e:
        print(f"Augmentation error: {str(e)}")
        return None, None

# =============================================
# STEP 5 - PREPROCESSING (NVIDIA DAVE-2 PIPELINE)
# =============================================
def preProcess(img):
    """
    Processes images for NVIDIA-style model input:
    1. Crop (remove sky/hood)
    2. Convert to YUV (like NVIDIA paper)
    3. Gaussian blur
    4. Resize to 200x66
    5. Normalize [0,1]
    """
    img = img[54:120, :, :]  # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    return img / 255.0

# =============================================
# STEP 6 - MODEL ARCHITECTURE (LIGHTWEIGHT FOR PI)
# =============================================
def createModel(input_shape=(66,200,3)):
    """
    Optimized NVIDIA DAVE-2 architecture with:
    - ELU activations (smoother than ReLU)
    - Dropout for regularization
    - Reduced parameters for Raspberry Pi
    """
    model = tf.keras.Sequential([
        Conv2D(24, (5,5), strides=(2,2), activation='elu', input_shape=input_shape),
        Conv2D(36, (5,5), strides=(2,2), activation='elu'),
        Conv2D(48, (5,5), strides=(2,2), activation='elu'),
        Conv2D(64, (3,3), activation='elu'),
        Conv2D(64, (3,3), activation='elu'),
        Flatten(),
        Dense(100, activation='elu'),
        Dropout(0.3),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()
    return model

# =============================================
# STEP 7 - DATA GENERATOR (MEMORY-EFFICIENT)
# =============================================
def dataGen(image_paths, steering_angles, batch_size, is_training):
    """
    Generator that yields batches of:
    - Augmented (if training) and preprocessed images
    - Corresponding steering angles
    """
    while True:
        img_batch = []
        steering_batch = []
        
        for _ in range(batch_size):
            idx = random.randint(0, len(image_paths)-1)
            
            if is_training:
                img, steering = augmentImage(image_paths[idx], steering_angles[idx])
            else:
                img = cv2.imread(image_paths[idx])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering = steering_angles[idx]
            
            if img is not None:
                img = preProcess(img)
                img_batch.append(img)
                steering_batch.append(steering)
        
        if len(img_batch) > 0:
            yield np.array(img_batch), np.array(steering_batch)