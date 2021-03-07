import math

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import initializers



def create_initializer(stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


class EmbeddingLayer(tf.keras.layers.Layer):
    def __int__(self,
                vocab_size=1,
                max_position_length=512,
                embedding_size=768,
                type_vocab_size=2,
                dropout_rate=0.2,
                stddev=0.02,
                **kwargs):
        super().__int__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater then 0."
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.stddev = stddev

        self.position_embedding = tf.keras.layers.Embedding(
            max_position_length,
            embedding_size,
            embeddings_initializer=create_initializer(stddev=stddev),
            name="position_embedding"
        )

        self.token_type_embedding = tf.keras.layers.Embedding(
            type_vocab_size,
            embedding_size,
            embeddings_initializer=create_initializer(stddev=stddev),
            name="token_type_embedding"
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        self.token_embedding = self.add_weight(
            name="token_embedding",
            shape=(self.vocab_size, self.embedding_size),
            initializer=create_initializer(stddev=self.stddev)
            )
    
    def call(self, input_ids, token_type_ids, training=None):
        seq_len = tf.shape(input_ids)[-1]
        position_ids = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.position_embedding(position_ids)

        token_type_embeddings = self.token_type_embedding(token_type_ids)
        
        token_embeddings = tf.matmul(input_ids, self.token_embedding)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        
        return embeddings


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self,
                hidden_size,
                num_attention_heads=1,
                attention_mask=None,
                attention_prods_dropout_prob=0.0,
                initializer_range=0.02,
                query_act=None,
                key_act=None,
                value_act=None,
                **kwargs):
        super().__init__(**kwargs)
        assert hidden_size % num_attention_heads == 0, "The hidden size (%d) is not a multiple of the number of \
            attention heads (%d)" % (hidden_size, num_attention_heads)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.initializer_range = initializer_range
        
        # query_layer, key_layer and value_layer, have the same inputs: [batch_size, sequence_length, hidden_size],
        # also have the same outputs: [batch_size, sequence_length, hidden_size]
        self.query_layer = tf.keras.layers.Dense(hidden_size, activation=query_act, 
                                                kernel_initializer=create_initializer(initializer_range))
        self.key_layer = tf.keras.layers.Dense(hidden_size, activation=key_act, 
                                                kernel_initializer=create_initializer(initializer_range))
        self.value_layer = tf.keras.layers.Dense(hidden_size, activation=value_act,
                                                kernel_intializer=create_initializer(initializer_range))
        
        # shape: [batch_size, sequence_length, sequence_length]
        self.attention_scores = tf.matmul(self.query_layer, self.key_layer, transpose_b=True)
        self.attention_scores = tf.multiply(self.attention_scores, 1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            attention_mask = 
















# class SimpleDense(Layer):
    
#     def __init__(self, units=32):
#         super(SimpleDense, self).__init__()
#         self.units = units
    
#     def build(self, input_shape):
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(input_shape[-1], self.units), dtype="float32"),
#             trainable=True)
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(self.units,), dtype="float32"),
#             trainable=True)
    
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b

# linear_layer = SimpleDense(4)

# y = linear_layer(tf.ones((2,2)))
# assert len(linear_layer.weights) == 2

# assert len(linear_layer.trainable_weights) == 2