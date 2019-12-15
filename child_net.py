import numpy as np 
import tensorflow as tf

import keras
from keras import models, layers, datasets, utils, backend, optimizers, initializers
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D

# Constructing Child Networks
class ChildNetwork:
    def __init__(self, x_train, y_train, x_test, y_test, sess, batch_size=200, epoch=10):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
        self.batch_size = batch_size
        self.num_classes = 10
        self.epochs = epoch
        self.data_augmentation = True

        self.model = self.build_model()

        self.sess = sess
        
        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=self.opt,
                    metrics=['accuracy'])
    def reinitialize(self, x, y):
        self.x_train = x
        self.y_train = y

        var_to_init = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='child')
        init_new_vars_op = tf.variables_initializer(var_to_init)
        self.sess.run(init_new_vars_op)
        #self.model.reset_states()

    def build_model(self):
        # https://keras.io/examples/cifar10_cnn/
        with tf.variable_scope("child",reuse=tf.AUTO_REUSE):
            model = models.Sequential()
            model.add(Conv2D(32, (3, 3), padding='same',
                            input_shape=self.x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes))
            model.add(Activation('softmax'))
        return model
    
    def train(self):
        self.model.fit(self.x_train, self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=0)
        
    
    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)

'''
x_train, y_train = DataIterator().data_and_label
x_test, y_test = DataIterator(test = True).data_and_label


'''