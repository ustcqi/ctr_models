#coding:utf8
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf

from deep_match import DeepMatch
from flags import *
from data_iterator import init_required_data, DataIterator

train_file = "data/dbpedia.train"
test_file = "data/dbpedia.test"

def main(params, train_file, test_file):
  label_dict, word_dict, word_freq_dict, train_data = init_required_data(train_file)
  print(len(word_dict))
  train_iterator = DataIterator(params, label_dict, word_dict, train_data)
  params["n_classes"] = len(label_dict)
  deep_match = DeepMatch(params)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  i = 1
  for (x, y, word_num) in train_iterator:
    # [input, label] = sess.run([deep_match.input, deep_match.label], feed_dict={deep_match.user_input : x, deep_match.label : y}) 
    [_, loss] = sess.run([deep_match.train_op, deep_match.loss], feed_dict={deep_match.user_input : x, deep_match.label : y, deep_match.word_num : word_num}) 
    if i % 100 == 0:
      print("step %d, loss %.6f" % (i, loss[0]))
    i += 1
  # test accuracy
  label_dict, word_dict, word_freq_dict, test_data = init_required_data(test_file)
  test_iterator = DataIterator(params, label_dict, word_dict, test_data)
  acc = 0
  batch_num = len(test_data) // params["batch_size"]
  for (x, y, word_num) in test_iterator:
    batch_acc = sess.run(deep_match.accuracy, feed_dict={deep_match.user_input : x, deep_match.label : y, deep_match.word_num : word_num})
    acc += batch_acc
  print(acc / batch_num)

if __name__ == "__main__":
  params = {"topk" : 10, "batch_size" : 500, "n_sampled" : 10}
  main(params, train_file, test_file)
  sys.exit(0)
