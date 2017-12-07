# coding: utf-8

import tensorflow as tf
import fc_inference
import numpy as np
from data_sets import DataSets,GFDataSet,PDataSet

INPUT_NODE = 4

def prediction(data_sets):
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE])
    y = tf.placeholder(dtype=tf.float32, shape=[None])

    y_ = fc_inference.chg_inference(x,None,None,False)

    cross_entroy_mean = tf.reduce_mean(tf.squared_difference(y,y_))

    test_x,test_y = data_sets.test_batchs()

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess: 
        tf.global_variables_initializer().run()
        model_file = tf.train.latest_checkpoint('./model/')
        saver.restore(sess,model_file)
        mse_final,out  = sess.run([cross_entroy_mean,y_],
                             feed_dict={x:np.squeeze(np.array(test_x[0:1])),
                                        y:np.squeeze(np.array(test_y[0:1]))
                                       }
                            )
        print mse_final
        print '---->'
        print out

        print '<----'
        print test_y[0:1]


def main(argv=None):
    data_set = PDataSet()
    data_set.calcu_etf('./data/out_dir/ag1606_20160104.csv')
    prediction(data_set)
    #test_x,test_y = data_set.test_batchs()
    #print np.array(test_x)
    #print np.array(test_y)


if __name__ == '__main__':
    tf.app.run()
