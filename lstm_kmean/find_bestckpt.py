# Evaluation of TripleNet Checkpoints using Clustering Accuracy

import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment

# Custom functions
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from constants import model_path

style.use('seaborn')

# Sets GPU
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# Clustering Accuracy Helper Function
# Credited to: https://github.com/k-han/DTC/blob/master/utils/util.py
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

# Evaluation script 

if __name__ == '__main__':

	# Defines model and dataset parameters
	n_channels  = 14
	n_feat      = 128
	batch_size  = 256
	test_batch_size  = 256
	n_classes   = 10

	# Loads dataset
	with open(f'{model_path}/eeg/image/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']

	# Creates batch datasets
	train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
	
	# Initialises model and Adam optimizer
	triplenet = TripleNet(n_classes=n_classes)	
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)

	# Sets up checkpoints and tracks best checkpoint performance
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/best_ckpt', max_to_keep=5000)
	best_ckpt_file = ''
	best_ckpt_acc  = 1e-15

	# Loops over all saved checkpoints
	for ckpt_file in tqdm(triplenet_ckptman.checkpoints):

		# Restores model weights from the checkpoint
		triplenet_ckpt.restore(ckpt_file)
		test_loss = tf.keras.metrics.Mean()
		test_acc  = tf.keras.metrics.SparseCategoricalAccuracy()

		feat_X  = np.array([])	# Embeddings
		feat_Y  = np.array([])	# Labels

		# Extraction of features from test data
		tq = tqdm(test_batch)

		for idx, (X, Y) in enumerate(tq, start=1):
			_, feat = triplenet(X, training=False)
			feat_X = np.concatenate((feat_X, feat.numpy()), axis=0) if feat_X.size else feat.numpy()
			feat_Y = np.concatenate((feat_Y, Y.numpy()), axis=0) if feat_Y.size else Y.numpy()

		# Random clustering
		kmeans = KMeans(n_clusters=n_classes,random_state=45)
		kmeans.fit(feat_X)
		labels = kmeans.labels_

		# Clustering accuracy
		kmeanacc = cluster_acc(feat_Y, labels)
		
		# Tracking of the best performing checkpoint
		if best_ckpt_acc < kmeanacc:
			best_ckpt_acc = kmeanacc
			best_ckpt_file = ckpt_file
		
		print('Checkpoint file: {}'.format(ckpt_file))
		print('Checkpoint test acc: {}'.format(kmeanacc))

	print('\n===============================================')
	print('Best acc file: {}'.format(best_ckpt_file))
	print('Best acc: {}'.format(best_ckpt_acc))
	print('===============================================\n')