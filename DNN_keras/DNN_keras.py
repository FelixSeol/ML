# -*- coding: iso-8859-15 -*-

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


def on_epoch_end(self, epoch, logs={}):
    val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
    val_targ = self.model.validation_data[1]
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print (" - val_f1: % f ? val_precision: % f - val_recall % f" % (_val_f1, _val_precision, _val_recall))
    return

dataset = np.loadtxt("creditcard.csv", delimiter=",");

X = dataset[:,0:28]
Y = dataset[:,28]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# create model
model = tf.keras.models.Sequential();
model.add(tf.keras.layers.Dense(24, input_dim=28, activation='relu'))
model.add(tf.keras.layers.Dense(12, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

metrics = Metrics()
# Fit the model
model.fit(X_train, Y_train,
        validation_data=(X_test,Y_test),
          epochs=10,
          batch_size=64,
          callbacks=[metrics])

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))