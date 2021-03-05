import token
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
                
                ):















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