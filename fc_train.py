# coding: utf-8

import tensorflow as tf
from data_sets import DataSets,GFDataSet
from monitor import Monitor
import numpy as np
import os
import fc_inference
#from sklearn.preprocessing import MinMaxScaler

INPUT_NODE = 13

LEARNING_RATE_BASE = 0.0 
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'
SUMMARY_DIR = 'flog/'

def train(data_sets):
    f_monitor = Monitor(SUMMARY_DIR)
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE])
    y = tf.placeholder(dtype=tf.float32, shape=[None])

    y_ = fc_inference.chg_inference(x, f_monitor, None, False)

    #存储训练轮数的变量
    global_step = tf.Variable(0,trainable=False)

    #variable_averages = tf.train.ExponentialMovingAverage(
    #    MOVING_AVERAGE_DECAY,global_step
    #    )

    #variable_averages_op = variable_averages.apply(
    #    tf.trainable_variables(
    #    ))


    cross_entroy_mean = tf.reduce_mean(tf.squared_difference(y,y_))

    f_monitor.scalar('cross_entroy_mean', cross_entroy_mean)


    #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #regular = fc_inference.chg_inference_regular(regularizer)

    #总损失为交叉熵和正则化损失的和
    #loss = cross_entroy_mean + regular
    loss = cross_entroy_mean
    f_monitor.scalar('loss', loss)

    train_x,train_y = data_sets.train_batchs()
    

    #learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
    #                                           global_step,
    #                                           len(train_x) * 2000,
    #                                           LEARNING_RATE_DECAY)

    train_step = tf.train.AdadeltaOptimizer(0.05).minimize(loss,global_step)

    #with tf.control_dependencies([train_step, variable_averages_op]):
    #    train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    merged = f_monitor.merged()
    with tf.Session() as sess:
        
        f_monitor.create(sess.graph)

        tf.global_variables_initializer().run()
        for i in range(2000):
            for step in range(len(train_x)):
                _,mse_final,summary,step_ = sess.run([train_step, loss, merged, global_step], 
                                                     feed_dict = {x:np.squeeze(np.array(train_x[step:step+1])),
                                                                  y:np.squeeze(np.array(train_y[step:step+1]))})
                f_monitor.writer(summary,step_)
                if step_ % 1000 == 0:
                    
                    print("After %d training step(s), loss on training"
                          " batch is %g acc" %(step_,mse_final))
                    saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=step_)
'''    
        i = 0
        train_batchs = data_sets.gf_train_batch()
        while i < data_sets.gf_train_count():
            data_set = train_batchs.next()
            j = 0
            while j < data_set.train_count() - 1:
                data_train = data_set.train_batch().next()
                batch_x = data_train[0]
                rebatch_x = np.reshape(batch_x,[-1, 13 * 4])
                batch_y = np.array([data_train[1]])
                _,mse_final,summary,step = sess.run([train_op, loss, merged, global_step], feed_dict={x:rebatch_x,y_:batch_y})
                if step  % 100 == 0:
                    print("After %d  training step(s), loss on training"
                         " batch is %g" %(step,mse_final))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
                f_monitor.writer(summary,step)
                j += 1
            i += 1
'''
def main(argv=None):
    data_set = GFDataSet()
    data_set.calcu_etf('./data/out_dir/ag1606_20160104.csv')
    train(data_set)
    #data_set = GFDataSet()
    #filename = './data/out_dir/ag1606_20160104.csv'
    #data_set.calcu_etf(filename)
    #data_sets = DataSets()
    #data_sets.gf_etf('./data/out_dir')
    #train(data_sets)

if __name__ == '__main__':
    tf.app.run()
