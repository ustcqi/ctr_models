import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_integer('user_input_length', 355, 'user single feature length')
# flags.DEFINE_integer('cross_input_length', 3, 'cross feature length')
flags.DEFINE_integer('user_input_length', 116, 'user single feature length')
flags.DEFINE_integer('cross_input_length', 239, 'cross feature length')
flags.DEFINE_integer('ad_input_length', 6, 'ad single feature length')
flags.DEFINE_float('learning_rate', 0.005, "learning rate")
flags.DEFINE_float('alpha', 2, 'loss weight for joint training')

flags.DEFINE_integer('vocab_size', 5000000, 'embedding table vocab size')
flags.DEFINE_integer('embedding_dim', 8, 'embedding dimension')

flags.DEFINE_integer('pred_layer1_units', 512, '')
flags.DEFINE_integer('pred_layer2_units', 256, '')
flags.DEFINE_integer('pred_layer3_units', 128, '')

flags.DEFINE_integer('user_layer1_units', 256, '')
flags.DEFINE_integer('user_layer2_units', 128, '')
flags.DEFINE_integer('ad_layer1_units', 256, '')
flags.DEFINE_integer('ad_layer2_units', 128, '')

flags.DEFINE_integer('match_output_units', 64, '')
