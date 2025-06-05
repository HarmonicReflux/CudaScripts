# code inspired by https://www.youtube.com/watch?v=HCLmM1PyDIs&ab_channel=JeffHeaton

import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)
devices = device_lib.list_local_devices()

# Get the list of physical devices available
gpus = tf.config.list_physical_devices('GPU')

# Set memory growth on all available GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

for d in devices:
    t = d.device_type
    name = d.physical_device_desc
    l = [item.split(':',1) for item in name.split(", ")]
    name_attr = dict([x for x in l if len(x)==2])
    dev = name_attr.get('name', 'Unnamed device')
    print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")


BATCH_SIZE = 16 #32
GPUS = ["GPU:0","GPU:1"]

def process(image, label):
    image = tf.image.resize(image, [299, 299]) / 255.0
    return image, label

strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync)

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(process).shuffle(500).batch(batch_size)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

EPOCHS = 5
LR = 0.001

tf.get_logger().setLevel('ERROR')

start = time.time()
with strategy.scope():
    model = tf.keras.applications.InceptionResNetV2(weights=None, classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=EPOCHS)

elapsed = time.time()-start
print (f'Training time: {hms_string(elapsed)}')

