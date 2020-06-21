#coding:utf8
import tensorflow as tf
from .flags import *

class DeepMR(object):
  
  def __init__(self, params=None):
    self._params = params
    self.build_model()

  # attention net, input : queries keys output : attention_output
  def attention(self, queries, keys, keys_length):
    """ 
      queries : [B H]
      keys : [B, T, H]
      keys_length: [B]
    """
    # query -> [B, T, H]
    [B, H] = queries.get_shape().as_list()
    [B, T, H] = keys.get_shape().as_list()
    # repeat query for T times at dimension 2, the same shape with keys
    queries = tf.tile(queries, [1, T]) 
    queries = tf.reshape(queries, [-1, T, H]) 
    interest_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    fc_layer_1 = tf.layers.dense(interest_input, 80, activation=tf.nn.sigmoid, name="fc_att1", reuse=tf.AUTO_REUSE)
    fc_layer_2 = tf.layers.dense(fc_layer_1, 40, activation=tf.nn.sigmoid, name="fc_att2", reuse=tf.AUTO_REUSE)
    fc_layer_3 = tf.layers.dense(fc_layer_2, 1, activation=None, name="fc_att3", reuse=tf.AUTO_REUSE)
    attention_output = tf.reshape(fc_layer_3, [-1, 1, T]) 
    keys_mask = tf.sequence_mask(keys_length, T)
    keys_mask = tf.expand_dims(keys_mask, 1)
    # - 2^32 主要是让这个神经元的值非常小,因为设成 0 对激活函数有要求
    paddings = tf.ones_like(attention_output) * (-2**32 + 1)
    attention_output = tf.where(keys_mask, attention_output, paddings)
    # scale 
    attention_output = attention_output / H ** 0.5 
    # activation
    attention_output = tf.nn.softmax(attention_output)
    # weighted sum B*1*T X B*T*H => B * 1 * H
    attention_output = tf.matmul(attention_output, keys)
    return attention_output
    # attention_output = tf.reshape(attention_output, [-1, H])

  def mask_input_embedding(self, sparse_input, embedding_input):
    embedding_mask = tf.cast(tf.greater(sparse_input, 0), tf.float32)
    embedding_mask = tf.expand_dims(embedding_mask, axis=2)
    embedding_mask = tf.tile(embedding_mask, (1, 1, FLAGS.embedding_dim))
    embedding_mask = tf.multiply(embedding_input, embedding_mask)
    return embedding_mask
  
  # params : {layer_size:[], activations:[], learning_rate:0.005, }
  def build_model(self):
    # instance_format: user_features ad_features cross_features
    self.user_input = tf.placeholder(tf.int32, shape=[None, FLAGS.user_input_length], name="user_input") 
    self.ad_input = tf.placeholder(tf.int32, shape=[None, FLAGS.ad_input_length], name="ad_input")
    self.cross_input = tf.placeholder(tf.int32, shape=[None, FLAGS.cross_input_length], name="cross_input")
    self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
  
    with tf.name_scope("embedding"):
      self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]), dtype=tf.float32, name="embedding")
      # self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]), dtype=tf.float32, name="embedding")
      
      self.user_embeddings = tf.nn.embedding_lookup(self.embedding, self.user_input, name="user_embeddings")
      self.ad_embeddings = tf.nn.embedding_lookup(self.embedding, self.ad_input, name="ad_embeddings")
      self.cross_embeddings = tf.nn.embedding_lookup(self.embedding, self.cross_input, name="cross_embeddings")

    with tf.name_scope("mask"):
      self.user_mask = self.mask_input_embedding(self.user_input, self.user_embeddings)
      self.ad_mask = self.mask_input_embedding(self.ad_input, self.ad_embeddings)
      self.cross_mask = self.mask_input_embedding(self.cross_input, self.cross_embeddings)

    with tf.name_scope("flatten"):
      self.user_flatten = tf.reshape(self.user_mask, [-1, FLAGS.user_input_length * FLAGS.embedding_dim]) 
      self.ad_flatten = tf.reshape(self.ad_mask, [-1, FLAGS.ad_input_length * FLAGS.embedding_dim])
      self.cross_flatten = tf.reshape(self.cross_mask, [-1, FLAGS.cross_input_length * FLAGS.embedding_dim])
    
    with tf.name_scope("match_net"):
      self.user_w1 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.user_input_length * FLAGS.embedding_dim, FLAGS.user_layer1_units)), name="match_user_w1")
      self.user_layer1 = tf.nn.relu(tf.matmul(self.user_flatten, self.user_w1))

      self.user_w2 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.user_layer1_units, FLAGS.user_layer2_units)), name="match_user_w2")
      self.user_layer2 = tf.nn.relu(tf.matmul(self.user_layer1, self.user_w2))

      self.user_w3 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.user_layer2_units, FLAGS.match_output_units)), name="match_user_w3")
      self.user_layer3 = tf.nn.tanh(tf.matmul(self.user_layer2, self.user_w3))
      
      self.ad_w1 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.ad_input_length * FLAGS.embedding_dim, FLAGS.ad_layer1_units)), name="match_ad_w1")
      self.ad_layer1 = tf.nn.relu(tf.matmul(self.ad_flatten, self.ad_w1))

      self.ad_w2 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.ad_layer1_units, FLAGS.ad_layer2_units)), name="match_ad_w2")
      self.ad_layer2 = tf.nn.relu(tf.matmul(self.ad_layer1, self.ad_w2))

      self.ad_w3 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.ad_layer2_units, FLAGS.match_output_units)), name="match_ad_w3")
      self.ad_layer3 = tf.nn.tanh(tf.matmul(self.ad_layer2, self.ad_w3))
      
      self.product = tf.multiply(self.user_layer3, self.ad_layer3)
    
      self.cosine_similarity = tf.reduce_sum(tf.multiply(self.user_layer3, self.ad_layer3), axis=1, keepdims=True)
      # self.match_score = tf.nn.sigmoid(self.cosine_similarity)

    with tf.name_scope("prediction_net"):
      self.pred_input = tf.concat([self.user_flatten, self.ad_flatten, self.cross_flatten, self.user_layer3, self.ad_layer3, self.product], axis=1)
      self.pred_input = tf.reshape(self.pred_input, [-1, (FLAGS.user_input_length + FLAGS.ad_input_length + FLAGS.cross_input_length) * FLAGS.embedding_dim + 3 * FLAGS.match_output_units])

      self.pred_w1 = tf.Variable(tf.glorot_uniform_initializer()( ((FLAGS.user_input_length + FLAGS.ad_input_length + FLAGS.cross_input_length) * FLAGS.embedding_dim + 3 * FLAGS.match_output_units, FLAGS.pred_layer1_units)), name="pred_w1")
      self.pred_layer1 = tf.nn.relu(tf.matmul(self.pred_input, self.pred_w1))

      self.pred_w2 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.pred_layer1_units, FLAGS.pred_layer2_units)), name="pred_w2")
      self.pred_layer2 = tf.nn.relu(tf.matmul(self.pred_layer1, self.pred_w2))

      self.pred_w3 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.pred_layer2_units, FLAGS.pred_layer3_units)), name="pred_w3")
      self.pred_layer3 = tf.nn.relu(tf.matmul(self.pred_layer2, self.pred_w3))

      self.pred_w4 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.pred_layer3_units, 1)), name="pred_w4")
      self.pred_output = tf.matmul(self.pred_layer3, self.pred_w4)
      
      self.pred_score = tf.nn.sigmoid(self.pred_output)

    with tf.name_scope("loss"):
      self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred_output, labels=self.label))

    with tf.name_scope("train"):
      self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
      self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
