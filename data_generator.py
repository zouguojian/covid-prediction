# -- coding: utf-8 --
import pandas as pd
import tensorflow as tf
import numpy as np

input_time_size=3
predict_time_size=3
window_size=1           #移动步长
city_name='武汉'
is_training=True

file='DXYArea.csv'
data = pd.read_csv(filepath_or_buffer=file, encoding='utf-8', usecols=[3, 10, 11, 12, 13, 14])
print(data.keys())

def re_data():  #return data set
    city_data=[]
    for line in data.values:             #loading the target city data
        if line[0]==city_name and len(line)==6:
            city_data.append([float(value) for value in line[1:-1]]+[float(char) for char in line[-1][:line[-1].index('.')].replace('-','').replace(':','').replace(' ','')])
    return city_data

def data_size():#return data size
    return len(re_data())

def generator():
    '''['cityName', 'city_confirmedCount', 'city_suspectedCount','city_curedCount', 'city_deadCount', 'updateTime']'''
    '''
    :return: yield the data of every time,
    shape:input_series:[input_time_size,features_size]
          label:[predict_time_size]
    '''

    city_data=re_data()
    city_data.reverse()
    low = 0

    while low < len(city_data) - (predict_time_size + input_time_size):
        yield (np.array(city_data)[low:low+input_time_size], np.array(city_data)[low+input_time_size:low+input_time_size+predict_time_size, 0])
        low += window_size
# for line,c in generator():
#     print(line.shape)

def next_batch(batch_size, epochs):
    '''
    :return the iterator!!!
    :param batch_size:
    :param epochs:
    :return:
    '''
    data_s=data_size()
    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
    if is_training:
        dataset = dataset.shuffle(buffer_size=(data_s - (predict_time_size + input_time_size))//window_size)
        dataset = dataset.repeat(count=epochs)
    dataset = dataset.batch(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

if __name__ == '__main__':
    print('____________________loading success_________________!!!')