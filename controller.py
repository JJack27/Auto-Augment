import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn



# Constructing Controller
class Controller:
    def __init__(self, session,
            baseline_acc,
            discount_factor = 0.95,
            ema_p = 0.05,
            lstm_units = 100, 
            operation_types = 16, 
            output_policies = 10,
            operation_prob = 11,
            operation_mag = 10,
            embedding_size = 20,
            save_model = False,
            load_model = False):
        
        # parameters for constructing controller's structure
        self._lstm_units = lstm_units
        self._output_policies = output_policies
        
        # attributes for embedding layer
        self._embedding_size = embedding_size
        self._embedding_weights = []

        # attributes for experiments
        self._operation_prob = operation_prob
        self._operation_types = operation_types
        self._operation_mag = operation_mag
        self._search_space_size = [self._operation_types, self._operation_prob, self._operation_mag]
        self.reward = tf.placeholder(tf.float32, ())
        self.discounted_reward = baseline_acc
        self.discount_factor = discount_factor
        self.baseline = baseline_acc
        self.ema_p = ema_p

        # model checkpoints 
        self._save_model = save_model
        self._load_model = load_model

        # dummy input placeholder for the tensorflow graph
        # generate random input
        # timesteps is set to one because we only have 1-d input
        self.x = tf.placeholder(dtype=tf.int32, shape=(1,None), name="input")
        self.input_x = np.zeros((1,1), dtype=np.int32)
        self.outputs = []
        self.softmaxes = []

        self.build_model()

        # session oriented
        self.sess = session
        self.initialized = False

    
    # build the model
    # for more info about keras.models: https://keras.io/models/model/
    # https://github.com/keras-team/keras/blob/master/keras/models.py
    # https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/engine/training.py#L28
    # =============================================
    # using pure tensorflow
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/dynamic_rnn
    # https://manutdzou.github.io/2017/11/27/tensorflow-lstm.html
    # https://www.oreilly.com/ideas/building-deep-learning-neural-networks-using-tensorflow-layers
    def build_model(self):
    
        # reuse trainable varaible of each layer to generate 5 policies
        with tf.variable_scope("controller", reuse=tf.AUTO_REUSE):
            # define LSTM layer
            self.lstm_layer = rnn.BasicLSTMCell(num_units=self._lstm_units, name="controller/rnn_lstm")
            h = np.random.uniform(-0.05, 0.05, [1,100])
            c = np.random.uniform(-0.05, 0.05, [1,100])
            #state_h = tf.Variable(initial_value=h, shape=[1, 100], dtype=np.float32)
            #state_c = tf.Variable(initial_value=c, shape=[1, 100], dtype=np.float32)
            #self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
            self.state = self.lstm_layer.zero_state(1, tf.float32)
            
            # define softmax classifier
            init = tf.contrib.layers.xavier_initializer()
            self.softmax_layer_type = tf.layers.Dense(self._operation_types, activation=tf.nn.softmax,  
                                  name="controller/softmax_layer_type", kernel_initializer=init)
            self.softmax_layer_prob = tf.layers.Dense(self._operation_prob, activation=tf.nn.softmax,  
                                  name="controller/softmax_layer_prob", kernel_initializer=init)
            self.softmax_layer_mag = tf.layers.Dense(self._operation_mag, activation=tf.nn.softmax, 
                                  name="controller/softmax_layer_mag", kernel_initializer=init)
            
            # initialized embedding layers 
            #with tf.variable_scope('embeddings'):
            for i in range(3):
                weights = tf.get_variable('state_embeddings_%d' % i,
                                shape=[self._search_space_size[i] + 1, self._embedding_size],
                                initializer=tf.initializers.random_uniform(-1., 1.))
                self._embedding_weights.append(weights)
             
            input_layer = tf.nn.embedding_lookup(self._embedding_weights[-1], self.x)
            

            # initialize LSTM states as initial_state
            
            #self.state = self.initial_state
            for i in range(self._output_policies):
                #with tf.name_scope('controller_output_%d' % i):
                    # generate operation type
                    output_type, self.state = tf.nn.dynamic_rnn(self.lstm_layer, 
                                            input_layer, 
                                            initial_state=self.state, 
                                            dtype=tf.float32)
                    softmax_result_type = self.softmax_layer_type(output_type)
                    pred_type = tf.argmax(softmax_result_type, axis=-1)
                    cell_input = tf.cast(pred_type, tf.int32)
                    cell_input = tf.add(cell_input, 1)
                    input_layer = tf.nn.embedding_lookup(self._embedding_weights[0], cell_input)

                    # generate operation probabilities
                    output_prob, self.state = tf.nn.dynamic_rnn(self.lstm_layer, 
                                            input_layer, 
                                            initial_state=self.state, 
                                            dtype=tf.float32)
                    softmax_result_prob = self.softmax_layer_prob(output_prob)
                    pred_prob = tf.argmax(softmax_result_prob, axis=-1)
                    cell_input = tf.cast(pred_prob, tf.int32)
                    cell_input = tf.add(cell_input, 1)
                    input_layer = tf.nn.embedding_lookup(self._embedding_weights[1], cell_input)
                    

                    # generate operation magnitude
                    output_mag, self.state = tf.nn.dynamic_rnn(self.lstm_layer, 
                                            input_layer, 
                                            initial_state=self.state, 
                                            dtype=tf.float32)
                    softmax_result_mag = self.softmax_layer_mag(output_mag)
                    pred_mag = tf.argmax(softmax_result_mag, axis=-1)
                    cell_input = tf.cast(pred_mag, tf.int32)
                    cell_input = tf.add(cell_input, 1)
                    input_layer = tf.nn.embedding_lookup(self._embedding_weights[2], cell_input)

                    self.softmaxes.append(softmax_result_type)
                    self.softmaxes.append(softmax_result_prob)
                    self.softmaxes.append(softmax_result_mag)

                    self.outputs.append(pred_type)
                    self.outputs.append(pred_prob)
                    self.outputs.append(pred_mag)

            #del cell_input, input_layer 
            #del pred_mag, pred_prob, pred_type
            #del output_mag, output_prob, output_type
            #del softmax_result_type, softmax_result_prob, softmax_result_mag

            # calculate gradient of trainable_variables w.r.t outputs from model
            #   - Thus, w.r.t the probabilities of each layer
            log_output = []
            for output in self.softmaxes:
                log_output.append(tf.math.log(output))
            var_to_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='controller')
            #var_to_train += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn')
            #for v in var_to_train:
            #    print(v)
            log_grad = tf.gradients(log_output, var_to_train)
            
            #####################################################
            self.gradients = [gradient * (self.reward) for gradient in log_grad]
            self.gradients = zip(self.gradients, var_to_train)
            self.optimizer = tf.train.GradientDescentOptimizer(0.0001).apply_gradients(self.gradients)
            return


    # method to update the parameters of lstm and softmax classifiers
    def update(self, accuracy):
        # Update trainable parameters via policy gradient
        #self._decay_rewards(accuracy)
        scale = (accuracy - self.baseline)
        #print("="*30)
        #print("Scale = %.4f" % scale)
        
        # initalize saver
        if self._save_model or self._load_model:
            saver = tf.train.Saver()

        if self._load_model:
            try:
                saver.restore(self.sess, "./model.ckpt")
            except:
                pass

        
                
        self.sess.run([self.optimizer], feed_dict={self.reward: scale, self.x: self.input_x})
        self._exp_moving_average_baseline(accuracy)
        if self._save_model:
            save_path = saver.save(self.sess, "./model.ckpt")
        
        return

    # given random input with self.embeddingsize, generate subpolicies
    def generate_subpolicies(self):

        # feed to model, get softmax
        #self.run_model(self.x)

        # run global variable initializer if needed
        if not self.initialized:
            print("variable initialized")
            self.sess.run(tf.global_variables_initializer())
            self.initialized = True

        # initalize saver
        if self._save_model or self._load_model:
            saver = tf.train.Saver() 
          
        if self._load_model:
            # if no model exists, skip
            try:
                saver.restore(self.sess, "./model.ckpt")
            except:
                pass
        
        # run session to get policies and softmaxes for inspection
        generated_subpolicies, soft = self.sess.run([self.outputs, self.softmaxes], feed_dict={self.x: self.input_x})
        print(soft[0])
        #self.input_x = generated_subpolicies[-1][-1].reshape(-1, 1)

        if self._save_model:
            save_path = saver.save(self.sess, "./model.ckpt")
         
        # parse softmaxes to subpolicies
        policies = []
        for i in range(self._output_policies):
            j = i * 3
            policy = [generated_subpolicies[j], generated_subpolicies[j + 1], generated_subpolicies[j + 2]]
            policies.append(policy)
        
        # format policy
        formated_policy = []
        for p in policies:
            operation, prob, mag = p
            formated_policy.append([operation[0][0], prob[0][0], mag[0][0]])
        
        print_policy(formated_policy)
        return formated_policy

    def _decay_rewards(self, accuracy):
        self.discounted_reward = self.discounted_reward * self.discount_factor + accuracy

    
    def _exp_moving_average_baseline(self, accuracy):
        self.baseline = self.ema_p * accuracy + (1 - self.ema_p) * self.baseline


        
def print_policy(policies):
    print("-"*30)
    for p in policies:
        operation, prob, mag = p
        print("Operation: %-6dProb: %-6dMag:%-6d" %(operation, prob, mag))
'''
tf.reset_default_graph()
sess = tf.Session()
controller = Controller(sess, 0.7)
po = controller.generate_subpolicies()
for i in [0.5, 0.4, 0.3, 0.2, 0.1]:
    controller.update(i)
    controller.generate_subpolicies()
'''