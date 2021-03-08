import math

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.keras.backend import dropout



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
        self.attention_prods_dropout_prob = attention_prods_dropout_prob
        self.initializer_range = initializer_range
        
        # query_layer, key_layer and value_layer, have the same inputs: [batch_size, seq_length, hidden_size],
        # also have the same outputs: [batch_size, seq_length, hidden_size]
        self.query_layer = tf.keras.layers.Dense(hidden_size, activation=query_act, 
                                                kernel_initializer=create_initializer(initializer_range))
        self.key_layer = tf.keras.layers.Dense(hidden_size, activation=key_act, 
                                                kernel_initializer=create_initializer(initializer_range))
        self.value_layer = tf.keras.layers.Dense(hidden_size, activation=value_act,
                                                kernel_intializer=create_initializer(initializer_range))

        # shape: [batch_size, seq_length, seq_length]
        self.attention_scores = tf.matmul(self.query_layer, self.key_layer, transpose_b=True)
        self.attention_scores = tf.multiply(self.attention_scores, 1.0 / math.sqrt(float(self.size_per_head)))

    def process_attention_mask(self, attention_mask):
        # attention_mask = [batch_size, seq_length]
        assert attention_mask.ndim == 2, "rank of attention mask must equal to 2, [batch_size, seq_length]"
        batch_size = attention_mask.shape[0]
        seq_length = attention_mask.shape[1]

        # attention_mask = [batch_size, 1, seq_length]
        attention_mask = tf.cast(tf.reshape(attention_mask, [batch_size, 1, seq_length]), tf.float32)
        # broadcast_ones = [batch_size, seq_length, 1]
        broadcast_ones = tf.ones(shape=[batch_size, seq_length, 1], dtype=tf.float32)
        # attention_mask = [batch_size, seq_length, seq_length]
        attention_mask = broadcast_ones * attention_mask

        return attention_mask
    
    def transpose_for_scores(self, inputs, batch_size, num_attention_heads, seq_length, size_per_head):
        # inputs = [batch_size, seq_length, hidden_size]
        assert inputs.shape[-1] == num_attention_heads * size_per_head, "The size of last dimension of inputs (%d) must equal to \
            num_attention_heads (%d) * size_per_head (%d)" % (inputs.shape[-1], num_attention_heads, size_per_head)
        
        # outputs = [batch_size, seq_length, num_attention_heads, size_per_head]
        outputs = tf.reshape(inputs, [batch_size, seq_length, num_attention_heads, size_per_head])
        # outputs = [batch_size, num_attention_heads, seq_length, size_per_head]
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        return outputs
    
    def call(self, inputs, attention_mask=None, is_training=True):
        # inputs is the output of embedding layer or the output of previous attention block
        # inputs = [batch_size, seq_length, hidden_size]
        assert inputs.ndim == 3, "rank of input_ids must equal to 3, [batch_size, seq_length, hidden_size]"
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]

        # query = [batch_size, seq_length, hidden_size]
        query = self.query_layer(inputs)
        # key = [batch_size, seq_length, hidden_size]
        key = self.key_layer(inputs)
        # value = [batch_size, seq_length, hidden_size]
        value = self.value_layer(inputs)

        # query = [batch_size, num_attention_heads, seq_length, size_per_head]
        query = self.transpose_for_scores(query, batch_size, self.num_attention_heads, seq_length, self.size_per_head)
        # key = [batch_size, num_attention_heads, seq_length, size_per_head]
        key = self.transpose_for_scores(key, batch_size, self.num_attention_heads, seq_length, self.size_per_head)

        # attention_scores = [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # attention_mask = [batch_size, seq_length, seq_length]
            attention_mask = self.process_attention_mask(attention_mask)
            # attention_mask = [batch_size, 1, seq_length, seq_length]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            attention_mask = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores += attention_mask
        
        # attention_probs = [batch_size, num_attention_heads, seq_length, seq_length]
        attention_probs = tf.keras.layers.Softmax()(attention_scores)
        attention_probs = tf.keras.layers.Dropout(self.attention_prods_dropout_prob)(attention_probs, training=is_training)

        # value = [batch_size, num_attention_heads, seq_length, size_per_head]
        value = self.transpose_for_scores(value, batch_size, self.num_attention_heads, seq_length, self.size_per_head)

        # outputs = [batch_size, num_attention_heads, seq_length, size_per_head]
        outputs = tf.matmul(attention_probs, value)
        # outputs = [batch_size, seq_length, num_attention_heads, size_per_head]
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        # outputs = [batch_size, seq_length, hidden_size]
        outputs = tf.reshape(outputs, [batch_size, seq_length, self.hidden_size])

        return outputs
        











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