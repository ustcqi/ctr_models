#coding:utf8
# import tensorflow as tf
import numpy as np
import math

# hahah... 赤裸的 dnn
class BareDNN(object):

  def __init__(self, params):
    self.params = params
    self.input_size = params['input_size']
    self.W1 = np.random.uniform(size=[params['input_size'], params['hidden1_units']])
    self.W2 = np.random.uniform(size=[params['hidden1_units'], 1])
    self.b1 = np.zeros(shape=[2, 1])
    self.b2 = np.zeros(shape=[1, 1])

  def sigmoid(self, x):
    return 1.0 /  (1. + np.exp(-x))

  def logloss(self, pred_y, true_y):
    return true_y * np.log(pred_y) + (1 - true_y) * np.log(1 - pred_y)

  def deriv_sigma(self, x):
    return self.sigmoid(x) * (1 - self.sigmoid(x))

  # x [1,2,3]
  def forward(self, x, y):
    # x [3, 1] W1 [3, 2] b1 [2 1] z1 [2, 1] h1 [2 1] z2 [1 1]
    self.z1 = np.matmul(np.transpose(self.W1), x) + self.b1
    self.h1 = self.sigmoid(self.z1)
    self.z2 = np.matmul(np.transpose(self.W2), self.h1) + self.b2
    self.pred_y = self.sigmoid(self.z2)
    print("pred_y:", self.pred_y)
    print("z2:", self.z2)
    print("h1:", self.h1)
    print("z1:", self.z1)
    self.loss = self.logloss(self.pred_y, y)

  def backward(self, x, y):
    learning_rate = self.params["learning_rate"]
    # deriv(loss, pred_y) [1 1]
    d_o = y / self.pred_y - (1 - y) / (1 - self.pred_y)

    # shape [1 1]
    d_z2 = np.multiply(d_o, self.deriv_sigma(self.z2))
    # shape [2 1]
    d_w2 = np.matmul(self.h1, d_z2)
    d_b2 = d_z2

    d_h1 = np.matmul(self.W2, d_z2)
    d_z1 = np.multiply(d_h1, self.deriv_sigma(self.z1))
    d_w1 = np.matmul(x, np.transpose(d_z1))

    d_b1 = d_z1
    """
    print('d_o:', d_o)
    print('d_z2:', d_z2)
    print('d_w2:', d_w2)
    print('d_b2:', d_b2)
    print('d_h1:', d_h1)
    print('d_z1:', d_z1)
    print('d_w1:',d_w1)
    print('d_b1:', d_b1)
    """
    # update weights
    print("W2:", self.W2)
    self.W2 -= learning_rate * d_w2
    print("after update W2:", self.W2)

    print("b2:", self.b2)
    self.b2 -= learning_rate * d_b2
    print("after update b2:", self.b2)

    print("W1:", self.W1)
    self.W1 -= learning_rate * d_w1
    print("after update W1:", self.W1)

    print("b1:", self.b1)
    self.b1 -= learning_rate * d_b1
    print("after update b1:", self.b1)

def train(model, X, Y):
  for epoch in range(1):
    for i in range(len(X)):
      x = np.reshape(X[i], (3, 1))
      y = Y[i]
      model.forward(x, y)
      model.backward(x, y)

def sigmoid(x):
  return 1.0 /  (1. + np.exp(-x))
  
def fake_data():
  x = np.random.random(size=[2, 3])

  w1 = np.random.uniform(size=[3, 2])
  b1 = np.random.uniform(size=[1, 2])
  z1 = np.matmul(x, w1) + b1
  h1 = sigmoid(z1)

  w2= np.random.uniform(size=[2, 1])
  b2 = np.random.uniform(size=[1, 1])
  z2 = np.matmul(h1, w2) + b2
  y = sigmoid(z2)
  y[y>=0.5] = 1
  y[y<0.5] = 0
  return x, y

def main():
  X, Y = fake_data() 
  params = {'input_size' : 3, 'hidden1_units' : 2, 'learning_rate' : 0.00001}
  dnn = BareDNN(params)
  train(dnn, X, Y) 

main()

