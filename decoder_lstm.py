# -- coding: utf-8 --
import tensorflow as tf
# import numpy as np
# import attention as attention
class lstm(object):
    def __init__(self,batch_size,predict_time,layer_num=1,nodes=128,is_training=True):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.is_training=is_training
        self.keep_pro()
        self.predict_time=predict_time
        self.out_num=1

    def keep_pro(self):
        '''
        used to define the self.keepProb value
        :return:
        '''
        if self.is_training:self.keepProb=0.5
        else:self.keepProb=1.0

    def lstm_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
        # if confirg.KeepProb<1:
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keepProb)

    def decoding(self):
        self.mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        self.initial_state=self.mlstm_cell.zero_state(self.batch_size,tf.float32)

    def predict(self,inputs):
        '''
        :param inputs: the shape is [batch_size, time_size, new_features].
        :param encoder_bi_lstm:
        :param encoder_init:
        :return: [batch_size,prediction_size]
        '''
        print('the inputs shape is :',inputs.get_shape().as_list())
        h = []
        h_state=inputs

        for i in range(self.predict_time):
            h_state = tf.reshape(h_state, shape=(-1, 1, self.nodes))
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                h_state, state = tf.nn.dynamic_rnn(cell=self.mlstm_cell, inputs=h_state, initial_state=self.initial_state,dtype=tf.float32)
                self.initial_state=state
            # we use the list h to recoder the out of decoder eatch time
            h.append(tf.reshape(h_state,shape=(self.batch_size,self.nodes)))

        #h=tf.convert_to_tensor(h,dtype=tf.float32)
        # h_state=tf.transpose(h,[1,0,2])
        h_state = tf.reshape(tf.transpose(h,[1,0,2]), [-1, self.nodes])

        #the full connect layer to output the end results
        with tf.variable_scope('Layer', reuse=tf.AUTO_REUSE):
            w = tf.get_variable("wight", [self.nodes, self.out_num],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("biases", [self.out_num],
                                   initializer=tf.constant_initializer(0))
            results = tf.matmul(h_state, w) + bias
        return tf.reshape(results, [self.batch_size, self.predict_time])