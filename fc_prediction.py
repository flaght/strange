# coding: utf-8

import tensorflow as tf
import fc_inference
import argparse
import csv
import numpy as np
from data_sets import DataSets, GFDataSet, PDataSet
import pdb
INPUT_NODE = 4 * 4

FLAGS = None

class Prediciton(object):
    def prediction(self, data_sets, time_step):
        x = tf.placeholder(dtype=tf.float32, shape=[None,time_step, INPUT_NODE])
        y = tf.placeholder(dtype=tf.float32, shape=[None, time_step, fc_inference.n_target])

        reshape_x = tf.reshape(x,[-1, INPUT_NODE])
        y_ = fc_inference.chg_inference(reshape_x, None, None, False)

        cross_entroy_mean = tf.reduce_mean(tf.square(tf.reshape(y_,[-1]) - tf.reshape(y, [-1])))

        test_x, test_y = data_sets.test_batch()

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            model_file = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, model_file)
            mse_final, out = sess.run([cross_entroy_mean, y_],
                                  feed_dict={x: test_x[0:1],
                                             y: test_y[0:1]
                                             }
                                  )
            return mse_final


def main(argv=None):
    
    # data_sets = DataSets()
    # data_sets.gf_etf('./data/temp')
    # while data_sets.is_range():
    #    data_set = data_sets.train_batch()
    #    pred = Prediciton()
    #    pred.prediction(data_set, data_set.test_step())
        # mse_final = prediction(data_set,data_set.test_step())
   #    print mse_final
   parser = argparse.ArgumentParser()
   parser.add_argument('--dir',type=str,help = 'data dir')
   FLAGS, unparsed = parser.parse_known_args()
   data_set = PDataSet()
   # data_set.calc_etf('./data/temp/ag1706_20161220.csv')
   data_set.calc_etf(FLAGS.dir)
   pred = Prediciton()
   loss = pred.prediction(data_set,data_set.test_step())
   with open("./result/prediction.csv","aw") as csvfile:
       writer = csv.writer(csvfile)
       writer.writerows([[loss,FLAGS.dir]])


if __name__ == '__main__':
    tf.app.run()
