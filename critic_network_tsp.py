import tensorflow as tf
import numpy as np
import tsp_env

def attention(W_ref, W_q, v, enc_outputs, query):
    with tf.variable_scope("attention_mask"):
        u_i0s = tf.einsum('kl,itl->itk', W_ref, enc_outputs)
        u_i1s = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
        u_is = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s))
        return tf.einsum('itk,it->ik', enc_outputs, tf.nn.softmax(u_is))

def critic_network(enc_inputs,
                   hidden_size=128, embedding_size=128,
                   max_time_steps=5, input_size=2,
                   batch_size=128,
                   initialization_stddev = 0.1,
                   n_processing_steps = 5, d = 128):
    with tf.variable_scope("critic"):
        # Embed inputs in larger dimensional tensors
        W_embed = tf.Variable(tf.random_normal([embedding_size, input_size],
                                               stddev=initialization_stddev))
        embedded_inputs = tf.einsum('kl,itl->itk', W_embed, enc_inputs)

        # Define encoder
        with tf.variable_scope("encoder"):
            enc_rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            enc_outputs, enc_final_state = tf.nn.dynamic_rnn(cell=enc_rnn_cell,
                                                             inputs=embedded_inputs,
                                                             dtype=tf.float32)
        # Define process block
        with tf.variable_scope("process_block"):
            process_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            first_process_block_input = tf.tile(tf.Variable(tf.random_normal([1, embedding_size]),
                                                            name='first_process_block_input'),
                                                [batch_size, 1])
            # Define attention weights
            with tf.variable_scope("attention_weights", reuse=True):
                W_ref = tf.Variable(tf.random_normal([embedding_size, embedding_size],
                                                     stddev=initialization_stddev),
                                    name='W_ref')
                W_q = tf.Variable(tf.random_normal([embedding_size, embedding_size],
                                                   stddev=initialization_stddev),
                                  name='W_q')
                v = tf.Variable(tf.random_normal([embedding_size], stddev=initialization_stddev),
                                name='v')

            # Processing chain
            processing_state = enc_final_state
            processing_input = first_process_block_input
            for t in range(n_processing_steps):
                processing_cell_output, processing_state = process_cell(inputs=processing_input,
                                                                       state=processing_state)
                processing_input = attention(W_ref, W_q, v,
                                             enc_outputs=enc_outputs, query=processing_cell_output)


        # Apply 2 layers of ReLu for decoding the processed state
        out = tf.squeeze(tf.layers.dense(inputs=tf.layers.dense(inputs=processing_cell_output,
                                                                units=d, activation=tf.nn.relu),
                                         units=1, activation=None))
    return out