# Sets up class indexing, data preprocessing and dataset loading functions

import numpy as np
import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
from functools import partial

# Custom model path
from constants import model_path

# Generates class name to index mappings ----------------------------------------------------------- 

data_cls = natsorted(glob(f'{model_path}/images/train/*'))

# Maps class name to indexe
cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
# Reverse mapping
idx2cls  = {value:key for key, value in cls2idx.items()}

# Preprocessing Function ----------------------------------------------------------- 

def preprocess_data(X, Y):
	X = tf.squeeze(X, axis=-1)
	max_val = tf.reduce_max(X)/2.0
	X = (X - max_val) / max_val		# Normalises signal
	X = tf.transpose(X, [1, 0])		# Transposes to time, feature
	X = tf.cast(X, dtype=tf.float32)
	Y = tf.argmax(Y)				# One-hot label converted to integer
	return X, Y

# Dataset Loader ----------------------------------------------------------- 

def load_complete_data(X, Y, batch_size=16):	
	dataset = tf.data.Dataset.from_tensor_slices((X, Y)).map(preprocess_data).shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset