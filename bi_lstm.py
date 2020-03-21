# -- coding: utf-8 --
import tensorflow as tf
#import numpy as np

class bi_lstm(object):
    def __init__(self,batch_size=1,hidden_dim=64,sequence_length=24,is_training=True):
        self.batch_size = batch_size  # batch size
        self.hidden_dim = hidden_dim  # hidden size
        self.sequence_length = sequence_length  # sequence length
        if is_training:self.output_keep_prob = 0.5  # to prevent overfit
        else:self.output_keep_prob=1.0

    def encoding(self):
        with tf.name_scope("rnn"):
            cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)  # single lstm unit
            self.cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.output_keep_prob)
            cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)  # single lstm unit
            self.cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.output_keep_prob)

        with tf.name_scope("output"):
            self.initial_bwstate=cell_bw.zero_state(batch_size=self.batch_size,dtype=tf.float32)
            self.initial_fwstate=cell_bw.zero_state(batch_size=self.batch_size,dtype=tf.float32)
    def extractor(self,inputs=None):
        '''

        :param inputs:
        :return: # the shape is [batch_size, hidden_size*2]
        '''
        with tf.variable_scope('encoder_bilstm', reuse=tf.AUTO_REUSE):
            self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(self.cell_bw, self.cell_fw, inputs,
                                                                        initial_state_fw=self.initial_fwstate,
                                                                        initial_state_bw=self.initial_bwstate,
                                                                        dtype=tf.float32)
        self.states=tf.reduce_sum([self.states[0][-1],self.states[1][-1]],axis=0)     #tensor shape is [batch, hidden_size*2]
        # print(self.states)
        # self.outputs=tf.concat(self.outputs,axis=2)   #tensor shape is [batch, sen_length,hidden_size*2]
        # print(self.states)
        # print(self.outputs)
        return self.states

        # return tf.transpose(tf.convert_to_tensor(value=re_state,dtype=tf.float32),perm=[1,0,2])

'''
if __name__=='__main__':
    data = np.random.random(size=[8, 48, 24,9])
    inputs = tf.placeholder(shape=[8, 48, 24,9], dtype=tf.float32, name='x')
    bi_lstm = bi_lstm(inputs, 8, 128, 48)
    state=bi_lstm.encoding()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict={inputs:data}
        print(sess.run(state,feed_dict=feed_dict).shape)
'''