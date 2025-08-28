import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    print("No GPU found, falling back to CPU.")
else:
    print(f"Found {len(gpus)} GPU(s).")

# Create a MirroredStrategy for multi-GPU support
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: ', strategy.num_replicas_in_sync)

# Create a simple model
def create_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset (using MNIST here as an example)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocessing the data (flattening the images)
train_images = train_images.reshape((-1, 784)).astype('float32') / 255.0
test_images = test_images.reshape((-1, 784)).astype('float32') / 255.0

# Create the model inside the strategy's scope
with strategy.scope():
    model = create_model()

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
