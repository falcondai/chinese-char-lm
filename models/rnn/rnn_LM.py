import tensorflow as tf
import fully_connected_DNN as DNN

class single_layer_RNN_LM(object):
    """single_layer_RNN_LM class is a wrapper of RNN based language model"""
    def __init__(self, hidden_unit, vocab_size, cell_type = "LSTM"):
        """
        constructor collects two model hyperparameters

        Arguments: 
        hidden_unit -- (int) the hidden unit size of RNN cell
        vocab_size -- (int) the size of the vocabulary 

        """
        super(single_layer_RNN_LM, self).__init__()

        self.hidden_unit = hidden_unit
        self.vocab_size = vocab_size
        self.DNN_mapping_func = DNN.fully_connected([self.vocab_size])
        self.cell_type = cell_type


    def RNN_LM(self, embedding_tensor, 
        target_ids, seq_len = None, reuse = False, variable_scope = "LSTM_LM"):
        """LM build the computational graph

        Arguments:
        embedding_tensor -- (tf.Variable, dtype = tf.float32) the input of LM, embedding_tensor.shape == (batch_size, steps, embedding_size) .
        target_ids -- (tf.Variable, type = tf.int32) the index of the prediction token in the vocabulary, target_ids < vocab_size self.
        seq_len -- (Optional)(tf.Variable or python list) if the input is masked, seq_len shall be provided, len(seq_len) == batch_size.
        reuse -- (Optional)(boolean) if the LM is reused
        variable_scope -- (Optional)(string) will raise error if reused is False and LM been called after the first
                          multiple times

        Return:
        perplexity -- (float) the perplexity of LM given the input_tensor and it's prediction outputs
        """
        with tf.variable_scope(variable_scope, initializer=tf.contrib.layers.xavier_initializer(dtype = tf.float32)) as scope:
            if reuse:
                scope.reuse_variables()

            cell = None
            if self.cell_type == "LSTM":
                cell = tf.contrib.rnn.LSTMCell(num_units = self.hidden_unit, 
                                                use_peepholes=False, 
                                                cell_clip=None, 
                                                initializer=tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                                                num_proj=None, 
                                                proj_clip=None, 
                                                num_unit_shards=None, 
                                                num_proj_shards=None, 
                                                forget_bias=1.0, 
                                                state_is_tuple=True, 
                                                activation=tf.tanh)

            elif self.cell_type == "GRU":
                cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_unit,
                                            input_size=None,
                                            activation=tf.tanh,
                                            reuse=None)

            elif self.cell_type == "Basic_RNN":
                tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_unit,
                                            input_size=None,
                                            activation=tf.tanh,
                                            reuse=None)
            else:
                raise ValueError('RNN cell type not included in this LM model')

            output, state = tf.nn.dynamic_rnn(cell = cell, 
                                              inputs = embedding_tensor[:,:-1,:], 
                                              sequence_length=seq_len, 
                                              initial_state=None, 
                                              dtype=tf.float32, 
                                              parallel_iterations=None, 
                                              swap_memory=False, 
                                              time_major=False, 
                                              scope=None)
            

            batch_size = embedding_tensor.get_shape()[0]
            max_length = embedding_tensor.get_shape()[1]

            output_reshape = tf.reshape(output, shape = [-1, self.hidden_unit])


            with tf.variable_scope("DNN_mapping_func") as scope_inner:
                if reuse:
                    scope_inner.reuse_variables()


                output_vector_flatten = self.DNN_mapping_func.build(input_tensor = output_reshape, 
                    reuse = reuse, variable_scope = scope_inner, dropout_rate=0.0, training_mode = True)

                output_vector_reshape_back = tf.reshape(output_vector_flatten, shape = tf.stack([batch_size, max_length - 1, self.vocab_size]))

                perplexity = tf.contrib.seq2seq.sequence_loss(logits = output_vector_reshape_back, 
                    targets = target_ids[:, 1:], weights = tf.ones_like(target_ids[:,1:], dtype = tf.float32))

        return perplexity