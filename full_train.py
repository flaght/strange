# coding: utf-8

import tensorflow as tf
from data_sets import DataSets
from monitor import Monitor
import numpy as np
import os
import full_inference
#from sklearn.preprocessing import MinMaxScaler

INPUT_NODE = 13 * 4
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'
SUMMARY_DIR = 'flog/'

def train(data_sets):
    f_monitor = Monitor(SUMMARY_DIR)
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None])

    y = full_inference.chg_inference(x, f_monitor)

    mse = tf.reduce_mean(tf.squared_difference(y,y_))
    f_monitor.scalar('mse', mse)

    train_step = tf.train.AdadeltaOptimizer(0.05).minimize(mse)

    saver = tf.train.Saver()

    merged = f_monitor.merged()
    with tf.Session() as sess:
        
        f_monitor.create(sess.graph)

        tf.global_variables_initializer().run()
        i = 0
        gloab_step = 0
        train_batchs = data_sets.gf_train_batch()
        while i < data_sets.gf_train_count():
            data_set = train_batchs.next()
            j = 0
            while j < data_set.train_count() - 1:
                data_train = data_set.train_batch().next()
                batch_x = data_train[0]
                rebatch_x = np.reshape(batch_x,[-1, 13 * 4])
                batch_y = np.array([data_train[1]])
                _,mse_final,summary  = sess.run([train_step, mse, merged], feed_dict={x:rebatch_x,y_:batch_y})
                if j % 10 == 0:
                    print("After[%d,%d] training step(s), loss on training"
                         " batch is %g" %(i,j,mse_final))
                f_monitor.writer(summary,gloab_step)
                j += 1
                gloab_step += 1
            i += 1
        saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME))

'''
        while i < data_set.train_count() - 1:
            data_train = data_set.train_batch().next()
            batch_x = data_train[0]
            rebatch_x = np.reshape(batch_x,[-1,13 * 4])
            batch_y = np.array([data_train[1]])
            _,mse_final = sess.run([train_step,mse], feed_dict={x:rebatch_x,y_:batch_y})
            if i % 10 == 0:
                print("After %d training step(s), loss on training"
                     " batch is %g" %(i,mse_final))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
            i += 1
'''
def main(argv=None):
    #data_set = GFDataSet()
    #filename = './data/out_dir/ag1606_20160104.csv'
    #data_set.calcu_etf(filename)
    data_sets = DataSets()
    data_sets.gf_etf('./data/out_dir')
    train(data_sets)

if __name__ == '__main__':
    tf.app.run()
