# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

dataset = np.loadtxt("creditcard.csv", delimiter=",");

X = dataset[:,0:28]
Y = dataset[:,28]

trainInput, testInput, trainOutput, testOutput= train_test_split(X, Y, test_size=0.2, random_state=0)

x = tf.placeholder(dtype=tf.float32, shape=[None, 28])
y = tf.placeholder(dtype=tf.float32)

W1 = tf.Variable(tf.random_uniform([28, 56], -1, 1.))
W2 = tf.Variable(tf.random_uniform([56, 28], -1, 1.))
W3 = tf.Variable(tf.random_uniform([28, 1], -1, 1.))

b1 = tf.Variable(tf.zeros([56]))
b2 = tf.Variable(tf.zeros([28]))
b3 = tf.Variable(tf.zeros([1]))

L1 = tf.add(tf.matmul(x, W1), b1)
L1 = tf.nn.relu(L1)
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)

model = tf.add(tf.matmul(L2, W3), b3)
model = tf.nn.sigmoid(model)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.transpose(model))
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
trainOp = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for step in range(100):
    sess.run(trainOp, feed_dict={x:trainInput, y:trainOutput})
    print(step+1, sess.run(cost, feed_dict={x:trainInput, y:trainOutput}))

prediction = tf.argmax(model, 1)
target = tf.argmax(y, 1)
print('예측값:', sess.run(prediction, feed_dict={x: testInput}))
print('실제값:', sess.run(target, feed_dict={y: testOutput}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={x: testInput, y: testOutput}))