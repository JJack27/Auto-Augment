from controller import Controller, print_policy
from child_net import ChildNetwork
from data_iterator import DataIterator

import tensorflow as tf 
import numpy as np 


# Train
CHILD_EPOCH = 20
CONTROLLER_EPOCH = 15000
accuracies = []
tf.reset_default_graph()

# this part is for colab. leaveing it is OK if running on your own computer
try:
    sess.close()
except:
    pass

sess = tf.Session()
x_test, y_test = DataIterator(test = True, shuffle=False).data_and_label
with sess.as_default():
    with sess.graph.as_default():

        # reload training dataset
        train_di = DataIterator(sampling=1)
        x_train, y_train = train_di.data_and_label
        
        # train the child network
        child = ChildNetwork(x_train, y_train, x_test, y_test, sess, 500, CHILD_EPOCH)
        child.train()

        # get accuracy
        loss, accuracy = child.evaluate()
        print("Baseline accuracy = %.4f" %(accuracy))
        del train_di

        # initialize controller
        controller = Controller(sess, accuracy)

        for i in range(CONTROLLER_EPOCH):
            # generate policies
            policy = controller.generate_subpolicies()
            train_di = DataIterator(sampling=1, policy=policy)
            x_train, y_train = train_di.data_and_label
            child.reinitialize(x_train, y_train)    # = ChildNetwork(x_train, y_train, x_test, y_test, sess, 500, CHILD_EPOCH)
            child.train()
            loss, accuracy = child.evaluate()
            print("Epoch = %d, accuracy = %.4f" %(i, accuracy))
            controller.update(accuracy)
            del train_di

