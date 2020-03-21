import tensorflow as tf
import numpy as np
import bi_lstm
import matplotlib.pyplot as plt
import decoder_lstm
import data_generator
import normalization as normalization
import os
tf.reset_default_graph()
#tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tf.reset_default_graph()
logs_path="board"

class parameter(object):
    def __init__(self):
        '''
        used to set the batch_size
        '''
        self.epochs=100              #the numbers of epoch size
        self.batch_size=8
        self.encoder_layer=1
        self.decoder_layer=1
        self.encoder_nodes=32
        self.learning_rate=0.001
        self.time_size=3            #the size of the input sequence
        self.prediction_size=3      #the size of predict sequence length
        self.features=18
        self.is_training=True
        self.window_step_size=1
        self.normalize=True          #the resouce data normalize
        self.target_city='宝山区'

'''
para used to set the parameters in later process
'''

class Model(object):
    def __init__(self,para):
        self.para=para
        self.x_input=tf.placeholder(dtype=tf.float32,shape=[None,self.para.time_size,self.para.features],name='pollutant')
        self.y=tf.placeholder(dtype=tf.float32,shape=[None,self.para.prediction_size])

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''

        '''
        feedforward and BN layer
        output shape:[batch, time_size,new_features]
        '''
        x_input=self.x_input
        # normal=normalization.Normalization(inputs=self.x_input,out_size=self.para.new_features,is_training=True)
        # normal.normal()
        # x_input = normal.feed_forward()
        # print('the output shape of normalization is :',x_input)
        # self.tag=x_input

        #this step use to encoding the input series data
        # encoder_init=Encoder.encoder(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # (c_state, h_state)=encoder_init.encoding()

        #this step, we try use Bi_lstm as spatial features extractor
        self.encoder_bi_lstm=bi_lstm.bi_lstm(self.para.batch_size,
                                             hidden_dim=self.para.encoder_nodes,
                                             sequence_length=self.para.time_size,
                                             is_training=self.para.is_training)
        self.encoder_bi_lstm.encoding()
        x_input=self.encoder_bi_lstm.extractor(x_input)
        # encoder_init=encodet_gru.gru(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # encoder_init=encoder_rnn.rnn(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        #self.encoder_init.encoding()

        #ready to compute the attention value
        # h_state = encoder_bi_lstm.extractor(inputs=)  # the shape is [batch_size, sequence_length , hidden_size*2]
        #
        # encoder_state=encoder_init.extractor(inputs=h_state)
        '''
            the above procession，is ABL-Learning Encoder and data process
            include:spatial feature extraction and time series feature extraction
            spatial h_state shape:[batch_size, sequence_length , hidden_size*2]
        '''

        #this step to presict the polutant concentration
        self.decoder_init=decoder_lstm.lstm(self.para.batch_size,
                                            self.para.prediction_size,
                                            self.para.decoder_layer,
                                            self.para.encoder_nodes,
                                            self.para.is_training)
        self.decoder_init.decoding() #init the layer and nodes
        self.pre=self.decoder_init.predict(x_input) #transfer parameter (input, bi-lstm object,lstm object)

    def train(self):
        '''
        :return:
        '''
        #pre, bi_alpha,l_alpha=self.model()
        self.cross_entropy = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pre), axis=0)), axis=0)
        tf.summary.scalar('cross_entropy',self.cross_entropy)
        # backprocess and update the parameters
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)
        return self.cross_entropy,self.train_op

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        #pre, bi_alpha, l_alpha= self.model()

        #return

    def accuracy(self,label,predict):
        '''
        :param Label: represents the observed value
        :param Predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        error = label - predict
        average_error = np.mean(np.fabs(error.astype(float)))
        print("In the test set, the prediction model mae error is : %.6f" % (average_error))

        rmse_error = np.sqrt(np.mean(np.square(np.array(label) - np.array(predict))))
        print("In the test set, prediction model rmse error is : %.6f" % (rmse_error))

        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
        print('In the test set, prediction model, the correlation coefficient is: %.6f' % (cor))
        return average_error,rmse_error,cor

    def describe(self,label,predict,prediction_size):
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:prediction_size], 'b*:', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:prediction_size], 'r*:', label=u'predicted value')
        # use the legend
        # plt.legend()
        plt.xlabel("time(days)", fontsize=17)
        plt.ylabel("people", fontsize=17)
        plt.title("the prediction of covid", fontsize=17)
        plt.show()

    def initialize_session(self):
        self.sess=tf.Session()
        self.saver=tf.train.Saver()

def re_current(A,max,min):
    return [round(float(num*(max-min)+min),3) for num in A]

def run_epoch(pre_model):
    '''
    from now on,the model begin to training, until the epoch to 100
    '''
    #saver = tf.train.Saver(max_to_keep=1)
    max_rmse = 100
    cross_entropy, train_op=pre_model.train()
    #with tf.Session() as sess:
    pre_model.sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

    '''init the generator!!!'''
    data_generator.is_training=True
    data_generator.city_name=pre_model.para.target_city
    data_generator.window_size=pre_model.para.window_step_size
    data_generator.input_time_size=pre_model.para.time_size
    data_generator.predict_time_size=pre_model.para.prediction_size

    next_elements=data_generator.next_batch(batch_size=pre_model.para.batch_size,epochs=pre_model.para.epochs)

    total_iter=(data_generator.data_size() - (data_generator.predict_time_size + data_generator.input_time_size))*pre_model.para.epochs \
               //(data_generator.window_size*pre_model.para.batch_size)
    print(total_iter)
    for i in range(total_iter):
        x, label =pre_model.sess.run(next_elements)
        # print('x shape is :',x.shape)
        summary, loss, _ = pre_model.sess.run((merged,cross_entropy,train_op), feed_dict={pre_model.x_input: x, pre_model.y: label})
        if i%100==0:
            print("After %d steps,the loss value is: %.6f" %(i,loss))
            writer.add_summary(summary, i%10)
            if max_rmse>loss:
                max_rmse=loss
                pre_model.saver.save(pre_model.sess,save_path='weights/covid.ckpt')

        # test process and save process
        #     rmse_error=evaluate(pre_model)
        #     if max_rmse>rmse_error:
        #         max_rmse=rmse_error
        #         pre_model.saver.save(pre_model.sess,save_path='weights/pollutant.ckpt')

def evaluate(pre_model):
    '''
    :param para:
    :param pre_model:
    :return:
    '''
    label_list = list()
    predict_list = list()

    #with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint('weights/')
    if not pre_model.para.is_training:
        print('the model weights has been loaded:')
        pre_model.saver.restore(pre_model.sess, model_file)

    pre_model.iterate.is_training = False
    pre_model.iterate.normalize = pre_model.para.normalize
    iterate_test = pre_model.iterate
    next_ = iterate_test.next_batch(batch_size=pre_model.para.batch_size, epochs=1)
    max,min=iterate_test.max_list[1],iterate_test.min_list[1]
    #print('the numbers of loop is :',(int(iterate.shape[0] * (1-iterate.data_divide)) - iterate.prediction_size)// (iterate.time_size * pre_model.para.batch_size))
    for i in range((int(iterate_test.shape[0] * (1-iterate_test.data_divide)) - iterate_test.prediction_size)//
                   (iterate_test.time_size * pre_model.para.batch_size)):
    #for i in range(2):
        # train process
        x, label = pre_model.sess.run(next_)
        #print('In the test step, the x shape is :',x.shape)
        pre_label,bi_alpha_, l_alpha_ ,w_= pre_model.sess.run((pre_model.pre, pre_model.bi_alpha, pre_model.l_alpha,pre_model.w),
                                                           feed_dict={pre_model.x_input: x})
        label_list.append(label)
        #print('the weights of decoder is :',w_)
        print('the predict value is :',re_current(pre_label[0],max,min))
        print('the real value is :',re_current(label[0],max,min))
        #print('the predict value is :', [round(float(num),2) for num in pre_label[0]])
        #print('the real value is :',label[0])
        predict_list.append(pre_label)

    label_list = np.array(re_current(np.reshape(np.array(label_list), [-1]),max,min))
    predict_list = np.array(re_current(np.reshape(np.array(predict_list), [-1]),max,min))

    average_Error, rmse_error, cor = pre_model.accuracy(label_list, predict_list)  #产生预测指标
    #pre_model.describe(label_list, predict_list, pre_model.para.prediction_size)   #预测值可视化
    return rmse_error

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    para = parameter()
    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    pre_model = Model(para)
    pre_model.model()
    pre_model.initialize_session()

    if int(val) == 1:para.is_training = True
    else:
        para.batch_size=32
        para.is_training = False

    if int(val) == 1:run_epoch(pre_model)
    else:evaluate(pre_model)

    print('finished____________________________finished_____________________________finished!!!')

if __name__ == '__main__':
    main()