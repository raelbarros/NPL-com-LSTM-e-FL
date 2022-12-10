import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam

class Model():
    def create_model(self):
        model = keras.Sequential([
            keras.layers.Embedding(input_dim=1920, output_dim=16),
            keras.layers.LSTM(64),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(),
            metrics=["accuracy"]
        )

        return model