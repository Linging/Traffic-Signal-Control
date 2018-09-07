import tensorflow as tf


def max_pool(x, kernel_size):
  """max pooling layer wrapper
  Args
    x:      4d tensor [batch, height, width, channels]
    kernel_size:   The size of the window for each dimension of the input tensor
  Returns
    a max pooling layer
  """
  return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, kernel_size[0], kernel_size[1], 1], padding='SAME')

def conv2d(x, n_kernel, kernel_size, stride=1):
  """convolutional layer with relu activation wrapper
  Args:
    x:          4d tensor [batch, height, width, channels]
    n_kernel:   number of kernels (output size)
    kernel_size:       2d array, kernel size. e.g. [8,8]
    stride:     stride
  Returns
    a conv2d layer
  """
  W = tf.Variable(tf.truncated_normal([kernel_size[0], kernel_size[1], int(x.get_shape()[3]), n_kernel], stddev=0.3))
  b = tf.Variable(tf.random_normal([n_kernel]))
  conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
  conv = tf.nn.bias_add(conv, b)
  return tf.nn.relu(conv)


def fc(x, n_output, activation_fn=None):
  """fully connected layer with relu activation wrapper
  Args
    x:          2d tensor [batch, n_input]
    n_output    output size
  """
  W=tf.Variable(tf.truncated_normal([int(x.get_shape()[1]), n_output], stddev=0.3))
  b=tf.Variable(tf.random_normal([n_output]))
  fc1 = tf.add(tf.matmul(x, W), b)
  if not activation_fn == None:
    fc1 = activation_fn(fc1)
  return fc1


def flatten(x):
  """flatten a 4d tensor into 2d
  Args
    x:          4d tensor [batch, height, width, channels]
  Returns a flattened 2d tensor
  """
  return tf.reshape(x, [-1, int(x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3])])


