# Evaluates the quality of the generated images using Tensorflow and a pretrained 2015 Inception model

import os
import os.path
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import sys
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# GPU initialization
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# Global setup and Inception model
MODEL_DIR = 'tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# This function computes the inception score for a list of images using the pretrained Inception model
# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)

  # Preprocesses images
  inps = []

  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))

  bs = 1  # Batch size

  with tf.Session() as sess:
    preds = []

    n_batches = int(math.ceil(float(len(inps)) / float(bs)))

    # Feeds images through Inception model
    for i in tqdm(range(n_batches)):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)

        # Model prediction
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)

    preds = np.concatenate(preds, 0)

    # Computes inception score over multiple splits
    scores = []

    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)

# Loads the pretrained Inception model and sets the global 'softmax' tensor
# This function is called automatically.
def _init_inception():
  global softmax

  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)

  # Extracts model
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

  # Load and parse Protobuf graph
  with tf.compat.v2.io.gfile.GFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  # Fetches and reshapes final feature
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()

    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []

            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)

            o.set_shape(tf.TensorShape(new_shape))

    # Rebuild softmax from weights        
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

# Initialise softmax if not imported
if softmax is None:
  _init_inception()


