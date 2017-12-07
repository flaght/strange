# coding: utf-8

import tensorflow as tf
INPUT_NODE = 4 * 4
neurons_layer1 = 1024
neurons_layer2 = 512
neurons_layer3 = 256
neurons_layer4 = 128
n_target = 1


def chg_inference_regular(regularizer):
    with tf.variable_scope("", reuse=True):
        w1 = tf.get_variable("layer1/weight",[INPUT_NODE,neurons_layer1])
        w2 = tf.get_variable("layer2/weight", [neurons_layer1,neurons_layer2])
        w3 = tf.get_variable("layer3/weight",[neurons_layer2, neurons_layer3])
        w4 = tf.get_variable("layer4/weight",[neurons_layer3,neurons_layer4])
    return regularizer(w1) + regularizer(w2) + regularizer(w3) + regularizer(w4)


def chg_nn_layer(input_tensor, input_dim, out_dim, layer_name, monitor, avg_class=None,reuse=False, act=tf.nn.relu):

    with tf.variable_scope(layer_name):
        weights = tf.get_variable("weight",[input_dim,out_dim],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if monitor != None:
            monitor.variable_summaries(layer_name + '/weights', weights)

        biases = tf.get_variable("biases",[out_dim],initializer=tf.constant_initializer(0.0))
        if monitor != None:
            monitor.variable_summaries(layer_name + '/biases', biases)

        if avg_class == None:
            peractive = tf.add(tf.matmul(input_tensor, weights),biases)
        else:
            peractive = tf.add(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))
        if monitor != None:
            monitor.variable_summaries(layer_name + '/per_activations', peractive)


        activations = act(peractive, name='activations')
        if monitor != None:
            monitor.variable_summaries(layer_name + '/activations', activations)
        return activations
  
def chg_inference(input_tensor, monitor,avg_class, reuse): #涨跌幅前向计算
    hidden1 = chg_nn_layer(input_tensor,INPUT_NODE,neurons_layer1,
                           'layer1',monitor,avg_class,reuse)

    hidden2 = chg_nn_layer(hidden1,neurons_layer1,neurons_layer2,
                           'layer2',monitor,avg_class,reuse)

    hidden3 = chg_nn_layer(hidden2,neurons_layer2,neurons_layer3,
                           'layer3',monitor,avg_class,reuse)

    hidden4 = chg_nn_layer(hidden3,neurons_layer3,neurons_layer4,
                           'layer4',monitor,avg_class,reuse)

    output = chg_nn_layer(hidden4,neurons_layer4,n_target,
                          'output',monitor,avg_class,reuse)
    return output



def test_chg_inference(input_tensor, monitor,avg_class, reuse): #涨跌幅前向计算
    hidden1 = chg_nn_layer(input_tensor,INPUT_NODE,neurons_layer1,
                           'test_layer1',monitor,avg_class,reuse)

    hidden2 = chg_nn_layer(hidden1,neurons_layer1,neurons_layer2,
                           'test_layer2',monitor,avg_class,reuse)

    hidden3 = chg_nn_layer(hidden2,neurons_layer2,neurons_layer3,
                           'test_layer3',monitor,avg_class,reuse)

    hidden4 = chg_nn_layer(hidden3,neurons_layer3,neurons_layer4,
                           'test_layer4',monitor,avg_class,reuse)

    output = chg_nn_layer(hidden4,neurons_layer4,n_target,
                          'test_output',monitor,avg_class,reuse)
    return output
