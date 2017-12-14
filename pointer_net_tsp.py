import tensorflow as tf
import numpy as np
import tsp_env

def attention_mask(W_ref, W_q, v, enc_outputs, query, already_played_actions=None,
                   already_played_penalty=1e6):
    with tf.variable_scope("attention_mask"):
        u_i0s = tf.einsum('kl,itl->itk', W_ref, enc_outputs)
        u_i1s = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
        u_is = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s)) - already_played_penalty * already_played_actions
        return u_is, tf.nn.softmax(u_is)

def pointer_network(enc_inputs, decoder_targets,
                    hidden_size=128, embedding_size=128,
                    max_time_steps=10, input_size=2,
                    batch_size=128,
                    initialization_stddev=0.1):
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
    # Define decoder
    with tf.variable_scope("decoder"):
        decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        first_decoder_input = tf.tile(tf.Variable(tf.random_normal([1, embedding_size]),
                                                  name='first_decoder_input'), [batch_size, 1])
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

        # Training chain
        paths_loss = 0
        loss = 0
        decoder_input = first_decoder_input
        decoder_state = enc_final_state
        already_played_actions = tf.zeros(shape=[batch_size, max_time_steps], dtype=tf.float32)
        decoder_inputs = [decoder_input]
        for t in range(max_time_steps):
            dec_cell_output, decoder_state = decoder_cell(inputs=decoder_input,
                                                          state=decoder_state)
            attn_logits, _ = attention_mask(W_ref, W_q, v, enc_outputs, dec_cell_output,
                                            already_played_actions=already_played_actions,
                                            already_played_penalty=1e6)
            paths_loss += tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets[:, t],
                                                                                    depth=max_time_steps),
                                                                  logits=attn_logits)
            loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets[:, t],
                                                                                            depth=max_time_steps),
                                                                          logits=attn_logits))
            # Teacher forcing of the next input
            decoder_input = tf.einsum('itk,it->ik', embedded_inputs,
                                      tf.one_hot(decoder_targets[:, t], depth=max_time_steps))
            decoder_inputs.append(decoder_input)
            already_played_actions += tf.one_hot(decoder_targets[:, t], depth=max_time_steps)

        # Inference chain
        decoder_input = first_decoder_input
        decoder_state = enc_final_state
        decoder_outputs = []
        already_played_actions = tf.zeros(shape=[batch_size, max_time_steps], dtype=tf.float32)
        for t in range(max_time_steps):
            dec_cell_output, decoder_state = decoder_cell(inputs=decoder_input,
                                                          state=decoder_state)
            _, attn_mask = attention_mask(W_ref, W_q, v, enc_outputs, dec_cell_output,
                                          already_played_actions=already_played_actions,
                                          already_played_penalty=1e6)
            decoder_outputs.append(tf.argmax(attn_mask, axis=1))
            decoder_input = tf.einsum('itk,it->ik', embedded_inputs, attn_mask)
            already_played_actions += tf.one_hot(decoder_outputs[-1],
                                                 depth=max_time_steps)

    return paths_loss, decoder_outputs, loss