#codingutf8
import sys
import string
import re

from flags import *
import numpy as np

train_file = "data/dbpedia.train"
test_file = "data/dbpedia.test"

# 生成 label 字典获取 label 个数
# 生成训练数据
# 统计 word 频次, 在计算 nce_loss 时可能会用到 sampled_values
def init_required_data(data_file):
  label_dict = {} # {
  label_cnt = 0
  word_dict = {}
  word_cnt = 1
  word_freq_dict = {}

  data = []
  with open(data_file) as in_file:
    for line in in_file:
      x = []
      items = line.strip('\n').split(' ')
      label = items[0].strip()
      words = items[1:]
      if label.find('__label__') != 0:
        continue
      if label not in label_dict:
        label_dict[label] = label_cnt
        label_cnt += 1
      y = label_dict[label]
      for word in words:
        word = word.replace("\n", "").replace("\t", "").replace("\r", "")
        word = re.sub(r'\\u.{4}', '', word.__repr__())
        word = ''.join(c for c in word if c not in string.punctuation)
        if word == "":
          continue
        if word not in  word_dict:
          word_dict[word] = word_cnt
          word_cnt += 1
        if word not in word_freq_dict:
          word_freq_dict[word] = 1
        else:
          word_freq_dict[word] += 1
        x.append(word)
      data.append((y, x))
      
  return label_dict, word_dict, word_freq_dict, data

# 过滤标点符号和非英文字符
class DataIterator(object):

  # self.label_dict, self.word_dict, self.word_freq_dict, self.data = init_required_data()
  def __init__(self, params, label_dict, word_dict, data):
    self.params = params
    self.batch_idx = 0
    self.data = data
    self.label_dict = label_dict
    self.word_dict = word_dict
    self.dsize = len(self.data)
    self.batch_num = self.dsize // self.params["batch_size"]

  def __iter__(self):
    return self

  def next(self):
    self.__next__()

  # return mini-batch train data
  def __next__(self):
    if self.batch_idx == self.batch_num:
      raise StopIteration
    params = self.params
    batch = self.data[self.batch_idx * params["batch_size"] : min((self.batch_idx + 1) * params["batch_size"], self.dsize)]
    batch_x = np.zeros((params["batch_size"], FLAGS.user_input_length), dtype=int)
    batch_y = []
    word_num = []
    i = 0
    for (y, x) in batch:
      j = 0
      for word in x:
        if word in self.word_dict:
          batch_x[i][j] = self.word_dict[word]
          j += 1
          if j >= FLAGS.user_input_length:
            break
      i += 1
      batch_y.append(y)
      word_num.append(j-1)

    self.batch_idx += 1
    return batch_x, np.array(batch_y).reshape(len(batch_y), 1), np.array(word_num).reshape(len(batch_y), 1)

if __name__ == "__main__":
  params = {"batch_size" : 32}
  for x, y in DataIterator(params, "./data/dbpedia.test"):
    # print(x.tolist())
    # print(y.tolist())
    pass
