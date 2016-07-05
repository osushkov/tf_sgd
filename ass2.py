import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import matplotlib.image as mpimg
import tensorflow as tf

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename


def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders


def load_letter(folder, min_images, image_size, pixel_depth):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)

    num_images = 0
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        try:
            image_data = (ndimage.imread(image_path).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception("unexpected shape: %s" % str(image_data.shape))
            dataset[num_images,:,:] = image_data
            num_images += 1
        except IOError as e:
            print("could not read: ", image_path)

    dataset = dataset[0:num_images,:,:]
    print dataset.shape
    print "mean: ", np.mean(dataset)
    print "std dev: ", np.std(dataset)
    # imgplot = plt.imshow(dataset[11])
    # raw_input("Press Enter to continue...")
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class, 28, 255.0)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def unpickle(pickle_files):
    labels = []
    data = []

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_data = pickle.load(f)
                labels.append(label)
                data.append(letter_data)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return labels, data

def test_split(train_labels, train_data, test_labels, test_data, test_ratio):
    unfolded_labels = np.empty(0, dtype=np.int32)
    unfolded_data = np.empty((0, train_data[0].shape[1], train_data[0].shape[2]), dtype=np.float32)

    for i in range(len(train_labels)):
        labels = np.full((train_data[i].shape[0]), train_labels[i], dtype=np.int32)
        unfolded_labels = np.concatenate((unfolded_labels, labels), axis=0)
        unfolded_data = np.concatenate((unfolded_data, train_data[i]), axis=0)

    for i in range(len(test_labels)):
        labels = np.full((test_data[i].shape[0]), test_labels[i], dtype=np.int32)
        unfolded_labels = np.concatenate((unfolded_labels, labels), axis=0)
        unfolded_data = np.concatenate((unfolded_data, test_data[i]), axis=0)

    print "unfolded labels: " + str(unfolded_labels.shape)
    print "unfolded shape:" + str(unfolded_data.shape)

    permutation = np.random.permutation(unfolded_labels.shape[0])
    shuffled_dataset = unfolded_data[permutation,:,:]
    shuffled_labels = unfolded_labels[permutation]

    test_length = int(shuffled_labels.shape[0] * test_ratio)

    test_labels = shuffled_labels[:test_length]
    test_data = shuffled_dataset[:test_length]

    train_labels = shuffled_labels[test_length:]
    train_data = shuffled_dataset[test_length:]

    return train_labels, train_data, test_labels, test_data

def generate_train_and_test():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    train_data_filenames = maybe_pickle(train_folders, 45000)
    test_data_filenames = maybe_pickle(test_folders, 1800)

    train_labels, train_data = unpickle(train_data_filenames)
    test_labels, test_data = unpickle(test_data_filenames)

    train_labels, train_data, test_labels, test_data = \
        test_split(train_labels, train_data, test_labels, test_data, 0.1)

    train_data.shape = (train_data.shape[0], train_data.shape[1] * train_data.shape[2])
    test_data.shape = (test_data.shape[0], test_data.shape[1] * test_data.shape[2])

    pickle_file = 'notMNIST.pickle'
    try:
        f = open(pickle_file, 'wb')
        save_data = {
            'train_labels': train_labels,
            'train_data': train_data,
            'test_labels': test_labels,
            'test_data': test_data,
        }
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    return train_labels, train_data, test_labels, test_data

def oneHotLabels(labels, num_classes):
    return (np.arange(num_classes) == labels[:, None]).astype(np.float32)


num_classes = 10
np.random.seed(133)
image_size = 28

pickle_file = 'notMNIST.pickle'
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        save_data = pickle.load(f)
        train_labels = save_data['train_labels']
        train_data = save_data['train_data']
        test_labels = save_data['test_labels']
        test_data = save_data['test_data']
        del save_data
else:
    train_labels, train_data, test_labels, test_data = generate_train_and_test()

train_labels = oneHotLabels(train_labels, num_classes)
test_labels = oneHotLabels(test_labels, num_classes)

batch_size = 256
graph = tf.Graph()

l1_size = 1024
l2_size = 512
l3_size = 256

with graph.as_default():

  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_classes))
  tf_test_dataset = tf.constant(test_data)

  keep_prob = tf.placeholder(tf.float32)

  weights_l1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, l1_size], stddev=(1.0/image_size)))
  biases_l1 = tf.Variable(tf.zeros([l1_size]))
  l1 = tf.nn.elu(tf.matmul(tf_train_dataset, weights_l1) + biases_l1)
  l1_drop = tf.nn.dropout(l1, keep_prob)

  weights_l2 = tf.Variable(tf.truncated_normal([l1_size, l2_size], stddev=(1.0/math.sqrt(l1_size))))
  biases_l2 = tf.Variable(tf.zeros([l2_size]))
  l2 = tf.nn.elu(tf.matmul(l1_drop, weights_l2) + biases_l2)
  l2_drop = tf.nn.dropout(l2, keep_prob)

  weights_l3 = tf.Variable(tf.truncated_normal([l2_size, l3_size], stddev=(1.0/math.sqrt(l2_size))))
  biases_l3 = tf.Variable(tf.zeros([l3_size]))
  l3 = tf.nn.elu(tf.matmul(l2_drop, weights_l3) + biases_l3)
  l3_drop = tf.nn.dropout(l3, keep_prob)


  weights_output = tf.Variable(tf.truncated_normal([l3_size, num_classes]))
  biases_output = tf.Variable(tf.zeros([num_classes]))

  logits = tf.matmul(l3_drop, weights_output) + biases_output
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) +
    0.1*tf.nn.l2_loss(weights_l1) + 0.1*tf.nn.l2_loss(weights_l2) + 0.1*tf.nn.l2_loss(weights_l3))

  optimizer = tf.train.AdamOptimizer().minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)

  l1_e = tf.nn.elu(tf.matmul(tf_test_dataset, weights_l1) + biases_l1)
  l1_e_drop = tf.nn.dropout(l1_e, keep_prob)

  l2_e = tf.nn.elu(tf.matmul(l1_e, weights_l2) + biases_l2)
  l2_e_drop = tf.nn.dropout(l2_e, keep_prob)

  l3_e = tf.nn.elu(tf.matmul(l2_e, weights_l3) + biases_l3)
  l3_e_drop = tf.nn.dropout(l3_e, keep_prob)

  test_prediction = tf.nn.softmax(tf.matmul(l3_e_drop, weights_output) + biases_output)

num_steps = 10000


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_data[offset:(offset + batch_size)]
    batch_labels = train_labels[offset:(offset + batch_size)]

    feed_dict = {
        tf_train_dataset : batch_data,
        tf_train_labels : batch_labels,
        keep_prob: 1.0,
    }

    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))

  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval({keep_prob: 1.0}), test_labels))
