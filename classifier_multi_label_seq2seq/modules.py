# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:55:52 2020

@author: cm
"""


import tensorflow as tf
from tensorflow.python.util import nest
from classifier_multi_label_seq2seq.hyperparameters import Hyperparamters as hp

dict_id2label, dict_label2id = hp.dict_id2label, hp.dict_label2id


def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one 
      so that it becomes the decoder inputs.      
    Args:
      inputs: A 3d tensor with shape of [N, T, C]    
    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1]), inputs[:, :-1]), 1)


def cell_lstm(lstm_hidden_size, is_training, scope='lstm', reuse=None):
    """
    A cell of LSTM
    """
    with tf.variable_scope(scope, reuse=reuse):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size)
        lstm_cell_drop = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.5 if is_training else 1)
    return lstm_cell_drop


def cell_attention_lstm(units, input_, _is_training):
    """
    A cell of attention
    """
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=units,
                                                               memory=input_)
    lstm_cell_ = tf.contrib.rnn.BasicLSTMCell(units)
    lstm_cell_drop = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_,
                                                   output_keep_prob=0.5 if _is_training else 1)
    cell_with_attetion = tf.contrib.seq2seq.AttentionWrapper(lstm_cell_drop,
                                                             attention_mechanism,
                                                             hp.lstm_hidden_size)
    return cell_with_attetion


def encoder(inputs, hidden_size, encoder_inputs_length, _is_training=True, bi_direction=True, scope="Encoder",
            reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T], dtype of int32.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors, whose shape is (N, T, E).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder   
        num_units = hidden_size
        if bi_direction:
            cell_forward = tf.contrib.rnn.MultiRNNCell(
                [cell_lstm(num_units, _is_training) for i in range(hp.num_layer_lstm_encode)])
            cell_backward = tf.contrib.rnn.MultiRNNCell(
                [cell_lstm(num_units, _is_training) for i in range(hp.num_layer_lstm_encode)])
            (output_forward, output_backword), (state_forward, state_backward) = tf.nn.bidirectional_dynamic_rnn(
                cell_forward,
                cell_backward,
                inputs,
                sequence_length=encoder_inputs_length,
                dtype=tf.float32)

            memory = tf.concat([output_forward, output_backword], 2)
            state_c = tf.concat([state_forward[2].c, state_backward[2].c], 1)
            state_h = tf.concat([state_forward[2].h, state_backward[2].h], 1)
            state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)


        else:
            cell = tf.contrib.rnn.MultiRNNCell(cell_lstm(num_units) * hp.num_encode_lstm)
            memory, state = tf.nn.bidirectional_dynamic_rnn(cell,
                                                            inputs,
                                                            dtype=tf.float32)
    return memory, state


def decoder(inputs, memory, encode_state, _is_training=True, scope="Decoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder
        vocab_size = len(dict_label2id)
        if _is_training:
            memory_ = memory
            encode_state = encode_state
            batch_size = hp.batch_size
        else:
            if hp.is_beam_search:
                memory_ = tf.contrib.seq2seq.tile_batch(memory, multiplier=hp.beam_size)
                encode_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, hp.beam_size),
                                                  encode_state)
                batch_size = tf.shape(memory)[0] * hp.beam_size
            else:
                memory_ = memory
                encode_state = encode_state
                batch_size = tf.shape(memory)[0]

        cell_with_attention = cell_attention_lstm(units=hp.lstm_hidden_size,
                                                  input_=memory_,
                                                  _is_training=_is_training)
        h_decode_initial = cell_with_attention.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
            cell_state=encode_state)
        output_layer = tf.layers.Dense(vocab_size,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        embedding = tf.get_variable('decoder_embedding', [vocab_size, hp.decoder_embedding_size])
        embedding = tf.concat((tf.zeros(shape=[1, hp.decoder_embedding_size]), embedding[1:, :]), 0)

        if _is_training:

            decoder_inputs = tf.nn.embedding_lookup(embedding, shift_by_one(inputs))

            targets_length = tf.count_nonzero(inputs, axis=1, dtype=tf.int32)

            max_target_sequence_length = tf.reduce_max(targets_length, name='max_target_len')
            mask = tf.sequence_mask(targets_length, max_target_sequence_length, dtype=tf.float32, name='masks')

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs,
                                                                sequence_length=targets_length,
                                                                time_major=False,
                                                                name='training_helper')

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_with_attention,
                                                               helper=training_helper,
                                                               initial_state=h_decode_initial,
                                                               output_layer=output_layer)

            outputs, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                            impute_finished=True,
                                                                                            maximum_iterations=hp.num_labels)  ##解码token的长度

        else:
            mask = tf.zeros(shape=[tf.shape(memory)[0], hp.decoder_embedding_size])
            start_tokens = tf.fill([tf.shape(memory)[0]], dict_label2id['E'])
            end_token = dict_label2id['S']
            if hp.is_beam_search:

                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell_with_attention,
                                                                         embedding=embedding,
                                                                         start_tokens=start_tokens,
                                                                         end_token=end_token,
                                                                         initial_state=h_decode_initial,
                                                                         beam_width=hp.beam_size,
                                                                         output_layer=output_layer)

            else:

                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                           start_tokens=start_tokens,
                                                                           end_token=end_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_with_attention,
                                                                    helper=decoding_helper,
                                                                    initial_state=h_decode_initial,
                                                                    output_layer=output_layer)

            outputs, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                                            maximum_iterations=hp.max_length)
    return outputs, final_state, mask, final_sequence_length


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor. 
    
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)


if __name__ == '__main__':
    print('Done')
