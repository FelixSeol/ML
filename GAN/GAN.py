
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

#dataset = np.loadtxt("creditcard.csv", delimiter=",")
#X_train, X_test, Y_train, Y_test = train_test_split(features, classes, test_size=0.2, random_state=0)
X_train = np.loadtxt("creditcard_only0.csv", delimiter=",")[:,0:28]
testset = np.loadtxt("testSet.csv", delimiter=",")
X_test = testset[:,0:28]
Y_test = testset[:,28]
train_columns = X_train.shape[0]
test_columns = testset.shape[0]

X = tf.placeholder(tf.float32, [None, 28]) # Feature = 28
Z = tf.placeholder(tf.float32, [None, 128]) # Noise Dimension = 128

# ********* G-Network (Hidden Node # = 256)
G_W1 = tf.Variable(tf.random_normal([128, 256], stddev=0.01))
G_W2 = tf.Variable(tf.random_normal([256, 28], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([256]))
G_b2 = tf.Variable(tf.zeros([28]))


def generator(noise_z): # 128 -> 256 -> 28
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.tanh(tf.matmul(hidden, G_W2) + G_b2)
    return output

# ********* D-Network (Hidden Node # = 256)
D_W1 = tf.Variable(tf.random_normal([28 , 112], stddev=0.01))
D_W2 = tf.Variable(tf.random_normal([112 , 1], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([112]))
D_b2 = tf.Variable(tf.zeros([1]))

def discriminator(inputs): # 28 -> 56 -> 28 -> 1
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output

# ********* Generation, Loss, Optimization
G = generator(Z)

#loss_D = -tf.reduce_mean(tf.log(discriminator(X)) + tf.log(1 - discriminator(G)))
#loss_G = -tf.reduce_mean(tf.log(discriminator(G)))
loss_D = tf.reduce_mean((discriminator(X)) + (1 - discriminator(G)))
loss_G = tf.reduce_mean((discriminator(G)))

train_D = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(-loss_D, var_list=[D_W1, D_b1, D_W2, D_b2])
train_G = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(-loss_G, var_list=[G_W1, G_b1, G_W2, G_b2])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ********* Training and Testing
noise_test = np.random.normal(size=(10, 128)) # 10 = Test Sample Size, 112 = Noise Dimension
epochs_competed = 0
epoch_num = 50
index_in_epoch = 0
batch_size = 100
test_truecase_num = 98.0
test_falsecase_num = test_columns - test_truecase_num
classify_rate = 0.9

for epoch in range(epoch_num): # Num. of Epoch
    for start in range(int(train_columns / batch_size)): # 100 = Batch Size
        start = index_in_epoch
        index_in_epoch += batch_size
        if index_in_epoch > train_columns :
            epochs_competed += 1
            start = 0
            index_in_epoch = batch_size
            assert batch_size <= train_columns
        end = index_in_epoch
        batch_xs = X_train[start:end]
        noise = np.random.normal(size=(100, 128))

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss : {:.4}'.format(loss_val_D),
          'G loss : {:.4}'.format(loss_val_G))
    if epoch == 0 or (epoch + 1) % 10 == 0: # 10 = Saving Period
        samples = sess.run(G, feed_dict={Z: noise_test})

output = sess.run(discriminator(X), feed_dict={X:X_test})
TP = 0.0
TN = 0.0
FP = 0.0
FN = 0.0
for i, _output in enumerate(output):
    #print("i = {:4d}   , class = {:.9f}   , real_class = {:9f}".format(i+1,float(_output),Y_test[i]))
    if(_output < classify_rate)[0] and Y_test[i]==1.0:
        TP += 1.0
    elif(_output < classify_rate)[0] and Y_test[i]==0.0:
        FP += 1.0
    elif (_output > classify_rate)[0] and Y_test[i] == 1.0:
        FN += 1.0
    elif (_output > classify_rate)[0] and Y_test[i] == 0.0:
        TN += 1.0


print("Attack data : ",test_truecase_num)
print("Normal data : ",test_falsecase_num)
print("TP : ",TP);
print("TN : ",TN);
print("FP : ",FP);
print("FN : ",FN);
print("Accuracy : ", (TP+TN)/test_columns)
print("Precision : ", TP/(TP+FP))
print("Recall : ",TP/(TP+FN))
