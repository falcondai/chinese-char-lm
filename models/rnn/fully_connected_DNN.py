import tensorflow as tf

class fully_connected(object):
    """docstring for fully_connected"""
    def __init__(self, DNN_setting):
        super(fully_connected, self).__init__()

        """
            DNN_setting: list of layer size
                         e.g. [200, 300, ... ,400]
        """
        self.DNN_setting = DNN_setting


    def build(self, input_tensor, reuse, variable_scope, dropout_rate=0.0, training_mode = False):
        with tf.variable_scope(variable_scope) as scope:
            if reuse:
                scope.reuse_variables()
            layer_list = []
            dropout_layer = tf.layers.dropout(input_tensor, rate=dropout_rate, 
                noise_shape=None, seed=None, training=training_mode, name=None)
            layer_list.append(dropout_layer)

            for i in xrange(len(self.DNN_setting)-1):
                layer_list.append(tf.layers.dense(
                    inputs = layer_list[i],
                    units = self.DNN_setting[i],
                    activation = tf.tanh,
                    kernel_initializer = tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                    # kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale, scope=None),
                    name = 'fully_connected_'+str(i),
                    reuse = reuse
                                                 )
                                 )
            layer_list.append(tf.layers.dense(
                inputs = layer_list[-1],
                units = self.DNN_setting[-1],
                activation = tf.tanh,
                kernel_initializer = tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                # kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale, scope=None),
                name = 'fully_connected_'+str(-1),
                reuse = reuse
                                             )
                             )
            return layer_list[-1]

