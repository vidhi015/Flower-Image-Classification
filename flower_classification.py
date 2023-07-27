!pip install gradio
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/e
xample_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,
untar=True)
data_dir = pathlib.Path(data_dir)
roses = list(data_dir.glob('roses/*'))
print(roses[0])
PIL.Image.open(str(roses[1]))
img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
 data_dir,
 validation_split=0.2,
 subset="training",
 seed=123,
 image_size=(img_height, img_width),
 batch_size=batch_size)
2
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
 data_dir,
 validation_split=0.2,
 subset="validation",
 seed=123,
 image_size=(img_height, img_width),
 batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
 for i in range(9):
 ax = plt.subplot(3, 3, i + 1)
 plt.imshow(images[i].numpy().astype("uint8"))
 plt.title(class_names[labels[i]])
 plt.axis("off")
num_classes = 5
model = Sequential([
 layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_
height, img_width, 3)),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(num_classes,activation='softmax')
])
model.compile(optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_l
ogits=True),
 metrics=['accuracy'])
3
epochs=10
history = model.fit(
 train_ds,
 validation_data=val_ds,
 epochs=epochs
)
def predict_image(img):
 img_3d=img.reshape(-1,180,180,3)
 prediction=model.predict(img_3d)[0]
 return {class_names[i]: float(prediction[i]) for i in range(5)}
image = gr.inputs.Image(shape=(180,180))
label = gr.outputs.Label(num_top_classes=5)
gr.Interface(fn=predict_image, inputs=image, outputs=label,capture_sess
ion=True).launch(debug='True')