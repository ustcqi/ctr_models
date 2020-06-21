#coding:utf8
import math
import tensorflow as tf
from flags import *

class DeepMatch(object):

  def __init__(self, params):
    self.build_model(params)

  def mask_input_embedding(self, sparse_input, embedding_input):
    embedding_mask = tf.cast(tf.greater(sparse_input, 0), tf.float32)
    embedding_mask = tf.expand_dims(embedding_mask, axis=2)
    embedding_mask = tf.tile(embedding_mask, (1, 1, FLAGS.embedding_dim))
    embedding_mask = tf.multiply(embedding_input, embedding_mask)
    return embedding_mask

  def build_model(self, params):
    self.user_input = tf.placeholder(tf.int32, shape=[None, FLAGS.user_input_length], name="user_input")
    self.word_num = tf.placeholder(tf.float32, shape=[None, 1], name="word_num")
    # self.hist_click_items = tf.placeholder(tf.int32, shape=[None, FLAGS.max_hist_length], name="hist_click_input")
    self.label = tf.placeholder(tf.int64, shape=[None, 1], name="label")

    with tf.name_scope("embedding"):
      self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim], -1.0, 1.0), dtype=tf.float32, name="embedding")

      self.user_input_embedding = tf.nn.embedding_lookup(self.embedding, self.user_input, name="user_input_embedding")
      # self.hist_click_items_embedding = tf.nn.embedding_lookup(self.embedding, self.hist_click_items, name="hist_click_embedding")
      
    with tf.name_scope("mask"):
      self.user_mask = self.mask_input_embedding(self.user_input, self.user_input_embedding)
      # self.hist_click_mask = self.mask_input_embedding(self.hist_click_items, self.hist_click_items_embedding)

    with tf.name_scope("flatten"):
      self.user_mask_flatten = tf.reshape(self.user_mask, [-1, FLAGS.user_input_length * FLAGS.embedding_dim])
      # self.hist_click_mask_flatten = tf.reshape(self.hist_click_mask, [-1, FLAGS.max_hist_length * FLAGS.embedding_dim])

    with tf.name_scope("input_after_embedding"):
      self.input = tf.div(self.user_mask_flatten, self.word_num)
      # self.input = tf.concat([self.user_mask_flatten, self.hist_click_mask_flatten], axis=1) 

    with tf.name_scope("net"):
      self.w1 = tf.Variable(tf.random_normal([FLAGS.user_input_length * FLAGS.embedding_dim, FLAGS.layer1_unit_num]) , name="w1")  
      self.bias1 = tf.Variable(tf.random_normal([FLAGS.layer1_unit_num]))
      self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.input, self.w1), self.bias1))

    with tf.name_scope("nce"):
      self.nce_weights = tf.Variable(tf.truncated_normal([params['n_classes'], FLAGS.layer1_unit_num], stddev=1.0 / math.sqrt(FLAGS.layer1_unit_num)), name="nce_weight")
      self.nce_bias = tf.Variable(tf.zeros([params['n_classes']]), name="nce_bias")

    with tf.name_scope("logits"):
      self.logits = tf.add(tf.matmul(self.layer1, tf.transpose(self.nce_weights)), self.nce_bias)
    
    with tf.name_scope("topk"):
      topk_values, topk_indices = tf.nn.top_k(self.logits, params["topk"])

    with tf.name_scope("loss"):
      # 可以自定义函数生成 sampled_values
      # sampled_values = (sampled_candidates, true_expecte_count, sampled_expected_count)
      self.loss = tf.nn.nce_loss(weights=self.nce_weights, biases=self.nce_bias, labels=self.label, inputs=self.layer1, num_sampled=params["n_sampled"], num_classes=params["n_classes"])
      self.cost = tf.reduce_sum(self.loss) / params["batch_size"]

    with tf.name_scope("train"):
      self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    with tf.name_scope("accuracy"):
      self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.label)
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
