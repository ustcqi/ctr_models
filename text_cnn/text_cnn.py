#coding:utf8
import tensorflow as tf

class TextCnn(object):

  def __init__(self, params):
    self.params = params
    self.build_model(params)

  def mask_input_embedding(self, sparse_input, embedding_input):
    # sparse_input: si1 si2 ... 0 0 0 [None, length]
    sparse_mask = tf.cast(tf.greater(sparse_input, 0), tf.float32)
    # [None, length, 1]
    sparse_mask = tf.expand_dims(sparse_mask, axis=1)
    # [None, length, embedding_dim]
    sparse_mask = tf.title(sparse_mask, [1, 1, FLAGS.embedding_dim])
    # embedding_input: [None, length, embedding_dim]
    embedding_mask = tf.multiply(sparse_mask, embedding_input)
    return embedding_mask

  def build_model(self, params):
    self.input = tf.placeholder(tf.int32, [None, FLAGS.max_length], name="input")
    self.label = tf.placeholder(tf.int32, [None, 1], name="label")

    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      self.embedding = tf.Variables(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim], -1, 1), dtype=tf.float32, name="embedding")
      self.input_embedding = tf.nn.embedding_lookup(self.embedding, self.input, name="input_embedding") 
      # [None, mask_length, embedding_dim]
      self.input_embedding = self.mask_input_embedding(self.input, self.input_embedding)
  
    with tf.name_scope("net"):
      # [None, max_length, embedding_dim] => [None, max_length, embedding_dim, 1]
      self.input_embedding = tf.expand_dims(self.input_embedding, -1)
      pooling_outputs = []
      filter_sizes = (FLAGS.filter_sizes).split(',')
      for filter_size in filter_sizes:
        filter_shape = [filter_size, FLAGS.embedding_dim, 1, FLAGS.filter_num]
	filters = tf.Variables(tf.random_uniform(filter_shape))
        conv = tf.nn.conv2d(self.input_embedding, filters, strides=[1, 1, 1, 1], padding="VALID") 

