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
try:
    sess.close()
except:
    pass

sess = tf.Session()
x_test, y_test = DataIterator(test = True, shuffle=False).data_and_label
with sess.as_default():
    with sess.graph.as_default():
        train_di = DataIterator(sampling=1)
        x_train, y_train = train_di.data_and_label
        
        child = ChildNetwork(x_train, y_train, x_test, y_test, sess, 500, CHILD_EPOCH)
        child.train()
        loss, accuracy = child.evaluate()
        print("Baseline accuracy = %.4f" %(accuracy))
        del train_di
        controller = Controller(sess, accuracy)

        for i in range(CONTROLLER_EPOCH):
            
            policy = controller.generate_subpolicies()
            #policy = [[1,1,1]]
            train_di = DataIterator(sampling=1, policy=policy)
            x_train, y_train = train_di.data_and_label
            child.reinitialize(x_train, y_train)    # = ChildNetwork(x_train, y_train, x_test, y_test, sess, 500, CHILD_EPOCH)
            child.train()
            loss, accuracy = child.evaluate()
            print("Epoch = %d, accuracy = %.4f" %(i, accuracy))
            for _ in range(10):
                controller.update(accuracy)
            del train_di

