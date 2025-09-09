print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from TrainingUtlis import *
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = '/home/pi/Desktop/car/src/DataCollected'
BATCH_SIZE = 100
EPOCHS = 30
INPUT_SHAPE = (66, 200, 3)

def main():
    # Step 1 - Load data
    data = importDataInfo(DATA_PATH)
    print("Raw data samples:", len(data))

    # Step 2 - Balance data
    data = balanceData(data, display=True)

    # Step 3 - Prepare images/steering
    imagesPath, steerings = loadData(DATA_PATH, data)

    # Step 4 - Stratified train/val split
    bins = np.digitize(steerings, np.linspace(-1, 1, 10))
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
    for train_idx, val_idx in split.split(imagesPath, bins):
        xTrain, xVal = imagesPath[train_idx], imagesPath[val_idx]
        yTrain, yVal = steerings[train_idx], steerings[val_idx]
    print(f"Training: {len(xTrain)}, Validation: {len(xVal)}")

    # Step 5 - Create model (updated)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(24, (5,5), strides=(2,2), activation='elu', input_shape=INPUT_SHAPE),
        tf.keras.layers.Conv2D(36, (5,5), strides=(2,2), activation='elu'),
        tf.keras.layers.Conv2D(48, (5,5), strides=(2,2), activation='elu'),
        tf.keras.layers.Conv2D(64, (3,3), activation='elu'),
        tf.keras.layers.Conv2D(64, (3,3), activation='elu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='elu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='elu'),
        tf.keras.layers.Dense(1)
    ])

    # Step 6 - Compile with dynamic LR
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=1000,
        decay_rate=0.9)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  loss='mse',
                  metrics=['mae'])

    # Step 7 - Train with early stopping
    history = model.fit(
        dataGen(xTrain, yTrain, BATCH_SIZE, 1),
        steps_per_epoch=len(xTrain)//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=dataGen(xVal, yVal, BATCH_SIZE, 0),
        validation_steps=len(xVal)//BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
    )

    # Step 8 - Save models
    model.save('model.keras')
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Saved Keras AND TFLite models")

    # Step 9 - Plot results
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Training')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('MAE')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()