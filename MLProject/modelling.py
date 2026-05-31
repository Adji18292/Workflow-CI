import os
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras
import sys
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

dataset_path = "./fruits_preprocessing"
print("Memuat dataset dari:", dataset_path)
dataset = tf.data.Dataset.load(dataset_path)

DATASET_SIZE = len(dataset)
train_size = int(0.8 * DATASET_SIZE)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

print("Memulai proses training Deep Learning di CI...")

model = models.Sequential([
    layers.InputLayer(input_shape=(100, 100, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(131, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training cepat (1 epoch) karena ini hanya untuk validasi CI workflow
history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)

# Mencatat metrik (mlflow run sudah memulai run secara otomatis, jadi bisa langsung log)
mlflow.log_metric("accuracy", history.history['accuracy'][-1])
mlflow.log_metric("loss", history.history['loss'][-1])

# Menyimpan model secara lokal agar bisa dijadikan artefak dan docker image
model_dir = "my_model_dir"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
    
print("Menyimpan model ke my_model_dir...")
mlflow.keras.save_model(model, model_dir)

# Upload ke DagsHub juga
mlflow.log_artifacts(model_dir, artifact_path="model")

print("Selesai!")
