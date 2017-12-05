# coding: utf-8

import tensorflow as tf
from monitor import Monitor
INPUT_NODE = 13 * 4
neurons_layer1 = 1024
neurons_layer2 = 512
neurons_layer3 = 256
neurons_layer4 = 128
n_target = 1

def chg_inference(input_tensor, monitor): #涨跌幅前向计算
    weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
    bias_initializer = tf.constant_initializer(0.0)

    with tf.variable_scope('layer1'):
        hidden1_weights = tf.Variable(weight_initializer([INPUT_NODE,neurons_layer1]))
        hidden1_biases = tf.Variable(bias_initializer([neurons_layer1]))
        hidden1 = tf.nn.relu(tf.add(tf.matmul(input_tensor,hidden1_weights),hidden1_biases))

        monitor.variable_summaries('layer1/weight',hidden1_weights)
        monitor.variable_summaries('layer1/biases',hidden1_biases)
        monitor.variable_summaries('layer1/hidden',hidden1)

    with tf.variable_scope('layer2'):
        hidden2_weights = tf.Variable(weight_initializer([neurons_layer1,neurons_layer2]))
        hidden2_biases = tf.Variable(bias_initializer([neurons_layer2]))
        hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1,hidden2_weights),hidden2_biases))

        monitor.variable_summaries('layer2/weight',hidden2_weights)
        monitor.variable_summaries('layer2/biases',hidden2_biases)
        monitor.variable_summaries('layer2/hidden',hidden2)

    with tf.variable_scope('layer3'):
        hidden3_weights = tf.Variable(weight_initializer([neurons_layer2,neurons_layer3]))
        hidden3_biases = tf.Variable(bias_initializer([neurons_layer3]))
        hidden3 = tf.nn.relu(tf.add(tf.matmul(hidden2,hidden3_weights),hidden3_biases))

        monitor.variable_summaries('layer3/weight',hidden3_weights)
        monitor.variable_summaries('layer3/biases',hidden3_biases)
        monitor.variable_summaries('layer3/hidden',hidden3)

    with tf.variable_scope('layer4'):
        hidden4_weights = tf.Variable(weight_initializer([neurons_layer3,neurons_layer4]))
        hidden4_biases = tf.Variable(bias_initializer([neurons_layer4]))
        hidden4 = tf.nn.relu(tf.add(tf.matmul(hidden3,hidden4_weights),hidden4_biases))

        monitor.variable_summaries('layer4/weight',hidden4_weights)
        monitor.variable_summaries('layer4/biases',hidden4_biases)
        monitor.variable_summaries('layer4/hidden',hidden4)

    with tf.variable_scope('output'):
        output_weights = tf.Variable(weight_initializer([neurons_layer4,n_target]))
        output_biases = tf.Variable(bias_initializer([n_target]))
        output = tf.transpose(tf.add(tf.matmul(hidden4,output_weights),output_biases))
        monitor.variable_summaries('output/weight',output_weights)
        monitor.variable_summaries('output/biases',output_biases)
        monitor.variable_summaries('output/hidden',output)
    return output
