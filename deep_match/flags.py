#coding:utf8
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('user_input_length', 1000, 'user single feature length')
flags.DEFINE_float('learning_rate', 0.005, "learning rate")

flags.DEFINE_integer('vocab_size', 1000000, 'embedding table vocab size')
flags.DEFINE_integer('embedding_dim', 128, 'embedding dimension')

flags.DEFINE_integer('layer1_unit_num', 128, '') 
