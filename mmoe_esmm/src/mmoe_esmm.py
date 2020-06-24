#coding:utf8
import tensorflow as tf
from .flags import *

# Entire Space Multi-Task model for post-click conversion rate prediction
# Modeling task relationships in multi-task learning with Multi-gate Mixture-of-Experts
class MMOE_ESMM(object):
  
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
    attention_output = tf.reshape(attention_output, [-1, H])
    return attention_output

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
    # hist_click_seq, hist_click_length
    self.hist_click_seq = tf.placeholder(tf.int32, shape=[None, FLAGS.max_hist_click_length], name="hist_click_seq")
    self.hist_click_length = tf.placeholder(tf.int32, shape=[None], name="hist_click_length")
    self.target_ad = tf.placeholder(tf.int32, shape=[None], name="target_ad")

    self.ctr_label = tf.placeholder(tf.float32, shape=[None, 1], name="ctr_label")
    self.cvr_label = tf.placeholder(tf.float32, shape=[None, 1], name="cvr_label")
  
    with tf.name_scope("embedding"):
      self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]), dtype=tf.float32, name="embedding")
      
      self.user_embeddings = tf.nn.embedding_lookup(self.embedding, self.user_input, name="user_embeddings")
      self.ad_embeddings = tf.nn.embedding_lookup(self.embedding, self.ad_input, name="ad_embeddings")
      self.cross_embeddings = tf.nn.embedding_lookup(self.embedding, self.cross_input, name="cross_embeddings")

      # shape: B T H
      self.hist_click_seq_embeddings = tf.nn.embedding_lookup(self.embedding, self.hist_click_seq, name="hist_click_seq_emb")
      # shape: B H
      self.target_ad_embedding = tf.nn.embedding_lookup(self.embedding, self.target_ad, name="target_ad_embedding")

    with tf.name_scope("mask"):
      self.user_mask = self.mask_input_embedding(self.user_input, self.user_embeddings)
      self.ad_mask = self.mask_input_embedding(self.ad_input, self.ad_embeddings)
      self.cross_mask = self.mask_input_embedding(self.cross_input, self.cross_embeddings)

    with tf.name_scope("attention"):
      self.ctr_attention_output = self.attention(self.target_ad_embedding, self.hist_click_seq_embeddings, self.hist_click_length)
      # ctr cvr 用的序列可能不同, 一个点击序列 一个转化序列
      self.cvr_attention_output = self.attention(self.target_ad_embedding, self.hist_click_seq_embeddings, self.hist_click_length)

    with tf.name_scope("flatten"):
      self.user_flatten = tf.reshape(self.user_mask, [-1, FLAGS.user_input_length * FLAGS.embedding_dim]) 
      self.ad_flatten = tf.reshape(self.ad_mask, [-1, FLAGS.ad_input_length * FLAGS.embedding_dim])
      self.cross_flatten = tf.reshape(self.cross_mask, [-1, FLAGS.cross_input_length * FLAGS.embedding_dim])

    with tf.name_scope("input"):
      self.input = tf.concat([self.user_flatten, self.ad_flatten, self.cross_flatten, self.ctr_attention_output], axis=1)
      self.input_length = (self.user_input_length + self.ad_input_length + self.cross_input_length + 1) * FLAGS.embedding_dim
      self.input = tf.reshape(self.ctr_input, [-1, self.input_length])

    with tf.name_scope("experts"):
      # 设置 3 个 expert, input * expert, [None, input_length] * [input_length, expert_units] => [None, expert_units]
      self.expert1 = tf.Variable(tf.glorot_uniform_initializer()((self.input_length, FLAGS.expert_units)), name="expert1")
      self.expert2 = tf.Variable(tf.glorot_uniform_initializer()((self.input_length, FLAGS.expert_units)), name="expert2")
      self.expert2 = tf.Variable(tf.glorot_uniform_initializer()((self.input_length, FLAGS.expert_units)), name="expert3")

    with tf.name_scope("gates"):
      # gate_num = task_num
      self.gate1 = tf.Variabel(tf.glorot_uniform_initializer()((self.input_length, FLAGS.expert_num)), name="gate1")
      self.gate2 = tf.Variabel(tf.glorot_uniform_initializer()((self.input_length, FLAGS.expert_num)), name="gate2")

    with tf.name_scope("mmoe"):
      # [None, input_length] * [input_length, expert_num] => [None, expert_num]
      self.gate1_output = tf.nn.softmax(tf.matmul(self.input, self.gate1), name="gate1_output")
      self.gate2_output = tf.nn.softmax(tf.matmul(self.input, self.gate2), name="gate2_output")
      # expand the dimension in dimension 1, [None, expert_num] => [None, 1, expert_num]
      self.expanded_gate1_output = tf.expand_dims(self.gate1_output, axis=1, name="expanded_gate1_output")
      # repeat expert_units times in dimension 1
      self.expanded_gate1_output = tf.tile(self.expanded_gate1_output, [1, FLAGS.expert_units, 1], name="expanded_repeat_gate1_output")

      self.expanded_gate2_output = tf.expand_dims(self.gate2_output, axis=1)
      self.expanded_gate2_output = tf.tile(self.expanded_gate2_output, [1, FLAGS.expert_units, 1], name="expanded_repeat_gate2_output")
     
      # [None, input_length] * [input_length, expert_units] => [None, expert_units]
      self.expert1_output = tf.nn.relu(tf.matmul(self.input, self.expert1), name="expert1_output")
      self.expert2_output = tf.nn.relu(tf.matmul(self.input, self.expert2), name="expert2_output")
      self.expert3_output = tf.nn.relu(tf.matmul(self.input, self.expert3), name="expert3_output")

      # [None, expert_num] multiply [None, expert_units] => [None, expert_units]
      self.expert1_gate1_output = tf.multiply(self.expanded_gate1_output, self.expert1_output)
      self.expert2_gate1_output = tf.multiply(self.expanded_gate1_output, self.expert2_output)
      self.expert3_gate1_output = tf.multiply(self.expanded_gate1_output, self.expert3_output)
      # [ [None, expert_units], [None, expert_units], [None, expert_units] ] => element_wise_dot [None, expert_units]
      self.task1_mmoe_output = tf.reduce_sum([self.expert1_gate1_output, self.expert2_gate1_output, self.expert3_gate1_output], axis=0, name="task1_mmoe_output")

      self.expert1_gate2_output = tf.multiply(self.expanded_gate2_output, self.expert1_output)
      self.expert2_gate2_output = tf.multiply(self.expanded_gate2_output, self.expert2_output)
      self.expert3_gate2_output = tf.multiply(self.expanded_gate2_output, self.expert3_output)
      self.task2_mmoe_output = tf.reduce_sum([self.expert1_gate1_output, self.expert2_gate1_output, self.expert3_gate1_output], axis=0, name="task2_mmoe_output")
    
    with tf.name_scope("ctr_net"):

      self.ctr_w1 = tf.Variable(tf.glorot_uniform_initializer()( (self.input_length, FLAGS.ctr_layer1_units)), name="ctr_w1")
      self.ctr_layer1 = tf.nn.relu(tf.matmul(self.ctr_input, self.ctr_w1))

      self.ctr_w2 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.ctr_layer1_units, FLAGS.ctr_layer2_units)), name="ctr_w2")
      self.ctr_layer2 = tf.nn.relu(tf.matmul(self.ctr_layer1, self.ctr_w2))

      self.ctr_w3 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.ctr_layer2_units, FLAGS.ctr_layer3_units)), name="ctr_w3")
      self.ctr_layer3 = tf.nn.relu(tf.matmul(self.ctr_layer2, self.ctr_w3))

      self.ctr_w4 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.ctr_layer3_units, 1)), name="ctr_w4")
      self.ctr_output = tf.matmul(self.ctr_layer3, self.ctr_w4)
      
      self.ctr_score = tf.nn.sigmoid(self.ctr_output)

    with tf.name_scope("cvr_net"):

      self.cvr_w1 = tf.Variable(tf.glorot_uniform_initializer()( (self.input_length, FLAGS.cvr_layer1_units)), name="cvr_w1")
      self.cvr_layer1 = tf.nn.relu(tf.matmul(self.cvr_input, self.cvr_w1))

      self.cvr_w2 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.cvr_layer1_units, FLAGS.cvr_layer2_units)), name="cvr_w2")
      self.cvr_layer2 = tf.nn.relu(tf.matmul(self.cvr_layer1, self.cvr_w2))

      self.cvr_w3 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.cvr_layer2_units, FLAGS.cvr_layer3_units)), name="cvr_w3")
      self.cvr_layer3 = tf.nn.relu(tf.matmul(self.cvr_layer2, self.cvr_w3))

      self.cvr_w4 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.cvr_layer3_units, 1)), name="cvr_w4")
      self.cvr_output = tf.matmul(self.cvr_layer3, self.cvr_w4)
      
      self.cvr_score = tf.nn.sigmoid(self.cvr_output)

    with tf.name_scope("loss"):
      self.ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.ctr_output, labels=self.ctr_label))
      self.ctcvr_score = self.ctr_score * self.cvr_score
      self.ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(labels=self.cvr_label, predictions=self.ctcvr_score))
      # paper alpha=1
      self.loss = self.ctr_loss + FLAGS.alpha * self.ctcvr_loss

    with tf.name_scope("train"):
      self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
