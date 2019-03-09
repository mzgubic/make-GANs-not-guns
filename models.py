import tensorflow as tf
import tensorflow.contrib.layers as layers


def Generator1D(input_noise, depth=3, n_units=20, name='MyGenerator1D'):
    
    with tf.variable_scope(name):
        
        layer = input_noise
        
        for _ in range(depth):
            layer = layers.relu(layer, n_units)
            
        output = 5*layers.linear(layer, 1)
    
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    
    return output, tf_vars


def Adversary(data, depth=3, n_units=10, name='MyAdversary'):
    
    with tf.variable_scope(name):
        
        layer = data
        
        for _ in range(depth):
            layer = layers.relu(layer, n_units)
        
        logits = layers.linear(layer, 2)
    
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    
    return logits, tf_vars
