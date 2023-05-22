import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

# Define constants
IMG_SIZE = 48
BATCH_SIZE = 32
NUM_CLASSES = 43
EPOCHS = 10
LEARNING_RATE = 0.0001

# Load the data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
meta_df = pd.read_csv('Meta.csv')

# Define the image data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# Create the training data generator
train_generator = datagen.flow_from_dataframe(
    train_df,
    directory='Train/',
    x_col='Path',
    y_col='ClassId',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Create the validation data generator
valid_generator = datagen.flow_from_dataframe(
    test_df,
    directory='Test/',
    x_col='Path',
    y_col='ClassId',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load the pre-trained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the layers in the base model
base_model.trainable = False

# Add the top layers to the model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // BATCH_SIZE
)

# Define endpoint for traffic sign detection
app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    # Get the image from the request
    image_data = request.json['image']
    # Decode the image from base64
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    # Resize the image to the required size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    # Convert the image to a numpy array
    image = np.array(image) / 255.0
    # Add an extra dimension to the image array
    image = np.expand_dims(image, axis=0)
    # Make predictions using the model
    predictions = model.predict(image)
    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    # Get the class name from the metadata
    class_name = meta_df[meta_df['ClassId'] == predicted_class]['SignName'].values[0]
    # Return the predicted class
    response = {'class_id': predicted_class, 'class_name': class_name}
    return jsonify(response)
