# Data preparation for GAN training

import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
import skimage.io as skio
from skimage.transform import resize
from tifffile import imwrite
import numpy as np
import cv2
import h5py
from functools import partial
import tensorflow_io as tfio
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import style

# Custom function to save figures
from evaluation.save_figure import save_figure, save_figure_condition

style.use('seaborn')

# Data preprocessing function
def preprocess_data(X, Y, P, resolution=128):
	X = tf.squeeze(X, axis=-1)					# Removes single channel dimension
	max_val = tf.reduce_max(X)/2.0
	X = (X - max_val) / max_val					# Normalises data
	X = tf.transpose(X, [1, 0])
	X = tf.cast(X, dtype=tf.float32)

	# Converts one-hot labels to class indexes
	Y = tf.argmax(Y)									

	# Loads images from path, resizes, and normalises them
	I = tf.image.decode_jpeg(tf.io.read_file(P), channels=3)
	I = tf.image.resize(I, (resolution, resolution))
	I = (tf.cast( I, dtype=tf.float32 ) - 127.5) / 127.5

	return X, Y, I

# Loads data for training and preprocessing
def load_complete_data(X, Y, P, batch_size=16, dataset_type='train'):	
	if dataset_type == 'train':
		dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).map(preprocess_data).shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	else:
		dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).map(preprocess_data).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset

# Converts tensors to back to RGB sized range and saves the plot
def show_batch_images(X, save_path, Y=None):
	X = np.clip(np.uint8(((X.numpy() * 0.5) + 0.5) * 255), 0, 255)
	col = 4
	row = X.shape[0] // col

	for r in range(row):
		for c in range(col):
			plt.subplot2grid((row, col), (r, c), rowspan=1, colspan=1)
			plt.grid('off')
			plt.axis('off')
			if Y is not None:
				plt.title('{}'.format(Y[r * col + c]))
			plt.imshow(X[r * col + c])


	plt.tight_layout()
	plt.savefig(save_path)
	plt.clf()
	plt.close()