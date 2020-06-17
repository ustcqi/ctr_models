#coding:utf8
import tensorflow as tf
# from cityhash import CityHash64

from .flags import *

# 参考这篇博客 如何用 SparseFeature 结合 Dataset 处理离散特征的
# https://blog.csdn.net/yujianmin1990/article/details/80384994
# dataset + estimator
# https://cloud.tencent.com/developer/article/1063010

# instance format:                                                 
class DataIterator(object):

  def __init__(self, params):
    self._params = params

  # uid user_features topic ad_features conv
  def decode_csv(self, record):
    # assert length
    records = tf.string_split([record], "\t")
    query_features = tf.string_split(records.values[1:2], ",")

    # user feature
    user_input = tf.string_to_number(query_features.values[:116], tf.int64)
    user_input = tf.mod(user_input, FLAGS.vocab_size)
    user_input = tf.reshape(user_input, [FLAGS.user_input_length])

    cross_input = tf.string_to_number(query_features.values[116:], tf.int64)
    cross_input = tf.mod(cross_input, FLAGS.vocab_size)
    cross_input = tf.reshape(cross_input, [FLAGS.cross_input_length])

    hist_click_seq = tf.string_to_number(query_features.values[116:121], tf.int64)
    hist_click_seq = tf.mod(hist_click_seq, FLAGS.vocab_size)
    hist_click_seq = tf.reshape(hist_click_seq, [FLAGS.max_hist_click_length])

    # 在样本中增加一列 fake data
    hist_click_length = tf.string_to_number(records.values[5], tf.int64)

    # ad topic feature
    doc_topic_features = tf.string_split(records.values[2:3], ",")
    doc_topic_features = tf.string_to_number(doc_topic_features.values, tf.int64)
    doc_topic_features = tf.mod(doc_topic_features, FLAGS.vocab_size)
    doc_topic_features = tf.reshape(doc_topic_features, [FLAGS.cross_input_length])

    # ad feature
    doc_features = tf.string_split(records.values[3:4], ",")
    # campaign shape: B 1
    target_ad = tf.string_to_number(doc_features.values[0], tf.int64)
    target_ad = tf.mod(target_ad, FLAGS.vocab_size)

    ad_input = tf.string_to_number(doc_features.values, tf.int64)
    ad_input = tf.mod(ad_input, FLAGS.vocab_size)
    ad_input = tf.reshape(ad_input, [FLAGS.ad_input_length])

    # label
    labels = tf.string_to_number(records.values[4:5], tf.float32)
    labels = tf.reshape(labels, [-1])

    pcxr = tf.string_to_number(records.values[-1], tf.float32)
    pcxr = tf.reshape(pcxr, [-1])
    # return query_features, doc_features, labels
    # return query_features, doc_topic_features, doc_features, labels
    # return user_input, cross_input, doc_features, labels
    return user_input, cross_input, ad_input, hist_click_seq, hist_click_length, target_ad, labels

  def input_fn(self, input_file, shuffle=True):
    params = self._params
    dataset = tf.data.TextLineDataset(input_file)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=params["shuffle_buffer_size"], seed=123)
    dataset = dataset.map(self.decode_csv, num_parallel_calls=params['num_parallel_calls']) \
                     .repeat(params["epoch"]) \
                     .batch(params["batch_size"])
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    iterator = dataset.make_initializable_iterator()
    return iterator

  def hdfs_input_fn(self, input_file, shuffle=True):
    params = self._params
    dataset = tf.data.TextLineDataset.list_files(input_file)
    dataset = dataset.interleave(lambda filename: (tf.data.TextLineDataset(filename)), cycle_length=64)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=params["shuffle_buffer_size"], seed=123)
    dataset = dataset.map(self.decode_csv, num_parallel_calls=params['num_parallel_calls']) \
                     .repeat(params["epoch"]) \
                     .batch(params["batch_size"])
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    iterator = dataset.make_initializable_iterator()
    return iterator
