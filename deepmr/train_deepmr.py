# coding:utf8
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from src.deepmr import DeepMR
from src.data_iterator import DataIterator
from src.flags import FLAGS

train_start_day = sys.argv[1]
train_end_day = sys.argv[2]

log_format = '%(asctime)s - %(levelname)s - %(message)s'
log_file = './log/' + 'cvr_training' +  train_start_day + '_' + train_end_day + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
e = 0

def eval(sess, deepmr, eval_part_list, eval_params):
  auc = 0.0  
  avg_loss = 0.0
  scores = np.zeros((1, 1))
  losses = np.zeros((1, 1))
  all_labels = np.zeros((1, 1))
  cnt = 0

  data_iterator = DataIterator(eval_params)
  for part in eval_part_list:
    iterator = data_iterator.input_fn(part, shuffle=False)
    sess.run(iterator.initializer)
    (user_features, topic_features, ad_features, labels) = iterator.get_next()
    while True:
      try:
        (batch_user_feature, batch_topic_feature, batch_ad_feature, batch_label) = sess.run([user_features, topic_features, ad_features, labels])
        pred_score, loss = sess.run([deepmr.pred_score, deepmr.loss], feed_dict={deepmr.user_input : batch_user_feature, deepmr.cross_input : batch_topic_feature, deepmr.ad_input : batch_ad_feature, deepmr.label : batch_label})
        scores = np.vstack([pred_score, scores])
        losses = np.vstack([loss, losses])
        cnt += len(scores)
        all_labels = np.vstack([batch_label, all_labels])
      except ValueError:
        continue
      except tf.errors.OutOfRangeError:
        break
      true_and_pred = np.concatenate((all_labels[:-1], scores[:-1]), axis=1)
  global e
  np.savetxt('./pred/predictions_' + str(e) + '.txt', true_and_pred, fmt='%7.6f')
  e += 1
  avg_loss = np.sum(losses[:-1], axis=0) / len(losses[:-1])
  # try ... except ..
  auc = roc_auc_score(all_labels[:-1], scores[:-1])
  return auc, avg_loss

def get_hdfs_parts(hdfs_root, start_day, end_day):
  train_part_list = []
  eval_part_list = []
  while start_day <= end_day:
    result = os.popen("/home/serving/hadoop_client/hadoop/bin/hadoop fs -ls %s/dt=%s" % (hdfs_root, start_day))
    part_list = [part.split(' ')[-1] for part in result.read().splitlines() if len(part.split(' ')) >= 8]
    
    if start_day == end_day:
      eval_part_list = part_list[-2:]
      train_part_list.extend(part_list[:-2])
    else:
      train_part_list.extend(part_list)
    start_day = datetime.strptime(start_day, "%Y%m%d") + timedelta(days=1)
    start_day = start_day.strftime("%Y%m%d")
  return train_part_list, eval_part_list

def train(train_part_list, eval_part_list, train_params, eval_params):
  data_iterator = DataIterator(train_params)
  deepmr = DeepMR(train_params)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    i = 1
    for epoch in range(train_params["train_epoch"]):
      for part in train_part_list:
        logging.info("training part %s" % part)
        iterator = data_iterator.hdfs_input_fn(part)
        sess.run(iterator.initializer)
        (user_features, topic_features, ad_features, labels) = iterator.get_next()
        while True:
          try:
            (batch_user_feature, batch_topic_feature, batch_ad_feature, batch_label) = sess.run([user_features, topic_features, ad_features, labels])
            sess.run(deepmr.train_step, feed_dict={deepmr.user_input : batch_user_feature, deepmr.cross_input : batch_topic_feature, deepmr.ad_input : batch_ad_feature, deepmr.label : batch_label})
          except tf.errors.OutOfRangeError:
            break
        if i % 10 == 0:
          auc, avg_loss = eval(sess, deepmr, eval_part_list, eval_params)
          logging.info("epoch %d, auc %.4f, loss %.4f" % (epoch, auc, avg_loss[0]))
        i += 1
    auc, avg_loss = eval(sess, deepmr, eval_part_list, eval_params)
    logging.info("epoch %d, auc %.4f, loss %.4f" % (epoch, auc, avg_loss[0]))

if __name__ == '__main__':
  train_params = {"shuffle_buffer_size" : 800000,
                  "num_parallel_calls" : 16,
                  "epoch" : 1,
                  "batch_size" : 64,
                  "train_epoch" : 1,
                }
  eval_params = {"shuffle_buffer_size" : 400000,
                 "num_parallel_calls" : 16,
                 "epoch" : 1,
                 "batch_size" : 128}
  logging.info("FLAGS : %s, train_params : %s" % (str(FLAGS), str(train_params)))

  
  hdfs_root = ""
  train_part_list, eval_part_list = get_hdfs_parts(hdfs_root, train_start_day, train_end_day)
  train(train_part_list, eval_part_list, train_params, eval_params)
  # model_path = "./model/deepmr.ckpt"
  # meta_path = "./model/deepmr.ckpt.meta"
  sys.exit(0)
