import os
import numpy as np
import tensorflow as tf
from word2vec.data import SkipGramDataSet

dataset = SkipGramDataSet(os.path.join(os.path.curdir, "./data/wiki_en.txt"))
VOCAB_SIZE = dataset.vocab_size
EMBEDDING_SIZE = 128
SKIP_WINDOW = 2
NUM_SAMPLED = 64
BATCH_SIZE = 32
WINDOW_SIZE = 2
TRAIN_STEPS = 10000
LEARNING_RATE = 0.1

class Word2Vec(object):

  def __init__(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.name_scope("inputs"):
        self.x = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32, name='x1')
        self.y = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32, name='x2')

      with tf.name_scope("layer1"):
        self.W1 = tf.Variable(
          tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1, 1, name='W1'),
          dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE]),
                              dtype=tf.float32, name='b1')
      hidden = tf.add(self.b1, tf.matmul(self.x, self.W1))

      with tf.name_scope("layer2"):
        self.W2 = tf.Variable(
          tf.random_uniform([EMBEDDING_SIZE, VOCAB_SIZE], -1, 1, name='W2'),
          dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([VOCAB_SIZE]),
                              dtype=tf.float32, name='b2')

      self.prediction = tf.nn.softmax(
        tf.add(tf.matmul(hidden, self.W2), self.b2), name='prediction')

      log = self.y * tf.log(self.prediction)


      self.loss = tf.reduce_mean(
        -tf.reduce_sum(log, reduction_indices=[1], keepdims=True))

      self.opt = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        self.loss)

  def _one_hot_input(self, dataset):
    features, labels = dataset.gen_batch_inputs(BATCH_SIZE, WINDOW_SIZE)
    f, l = [], []
    for w in features:
      tmp = np.zeros([VOCAB_SIZE])
      tmp[w] = 1
      f.append(tmp)
    for w in labels:
      tmp = np.zeros(VOCAB_SIZE)
      tmp[w] = 1
      l.append(tmp)
    return f, l

  def train(self, dataset, n_iters, ):
    with tf.Session(graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.save(sess, "./text_data/text_data.ckpt")
      for i in range(n_iters):
        features, labels = self._one_hot_input(dataset)


        predi, loss = sess.run([self.prediction, self.loss],
                               feed_dict={
                                 self.x: features,
                                 self.y: labels
                               })
        print("loss:%s" % loss)

  def restore_model_ckpt(self):
    with tf.Session(graph=self.graph) as sess:
      # if word in dataset.data:
      #   print(dataset.data)
      saver = tf.train.Saver()
      saver.restore(sess, "model/model.ckpt")
      summary_writer = tf.summary.FileWriter('./log/', sess.graph)
      tmp = np.zeros([VOCAB_SIZE],  dtype='float32')
      tmp[2] = 1
      embedding = np.matmul(tmp, sess.run(self.W1))
      print(sess.run(self.W1))
      print(embedding.shape)

  def predict(self):
    pass

  def nearest(self, n):
    pass

  def similarity(self, a, b):
    pass

if __name__ == "__main__":
  word2vec = Word2Vec()
  word2vec.train(dataset, TRAIN_STEPS)
  word2vec.restore_model_ckpt()

