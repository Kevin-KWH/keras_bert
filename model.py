import json
import math
import copy


import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, Softmax, LayerNormalization
from tensorflow.keras import Model

MAX_PREDICTIONS_PER_SEQ = 2


def create_initializer(stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)

def check_input_mask(input_mask):
    unique_values, _ = tf.unique(tf.reshape(input_mask, [-1]))
    unique_values = list(unique_values.numpy())
    unique_values.sort()
    if len(unique_values) == 1 and unique_values[0] in [0, 1]:
        return True
    elif len(unique_values) == 2 and unique_values == [0, 1]:
        return True
    else:
        return False

def check_token_type_ids(token_type_ids, type_vocab_size):
    unique_values, _ = tf.unique(tf.reshape(token_type_ids, [-1]))
    unique_values = list(unique_values.numpy())
    return False if len(unique_values) > type_vocab_size else True

class BertConfig(object):
    def __init__(self,
                vocab_size=1,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02
                ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
    
    @classmethod
    def from_dict(cls, dict):
        config = BertConfig()
        for key, value in dict.items():
            config[key] = value
        return config
    
    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, "r") as fr:
            text = fr.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
    def get_config(self):
        config = {"vocab_size": self.vocab_size,
                  "hidden_size": self.hidden_size,
                  "num_hidden_layers": self.num_hidden_layers,
                  "num_attention_heads": self.num_attention_heads,
                  "intermediate_size": self.intermediate_size,
                  "hidden_act": self.hidden_act,
                  "hidden_dropout_prob": self.hidden_dropout_prob,
                  "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                  "max_position_embeddings": self.max_position_embeddings,
                  "type_vocab_size": self.type_vocab_size,
                  "initializer_range": self.initializer_range}
        return config


class EmbeddingLayer(Layer):
    def __init__(self,
                vocab_size=1,
                max_position_length=512,
                embedding_size=768,
                type_vocab_size=2,
                dropout_prob=0.2,
                stddev=0.02,
                name="embedding_layer",
                **kwargs):
        super(EmbeddingLayer, self).__init__()

        assert vocab_size > 0, "vocab_size must greater then 0."
        self.vocab_size = vocab_size
        self.max_position_length = max_position_length
        self.embedding_size = embedding_size
        self.type_vocab_size = type_vocab_size
        self.dropout_prob = dropout_prob
        self.stddev = stddev
        

        self.position_embedding = Embedding(
            max_position_length,
            embedding_size,
            embeddings_initializer=create_initializer(stddev=stddev),
            name="position_embedding_layer"
        )

        self.token_type_embedding = Embedding(
            type_vocab_size,
            embedding_size,
            embeddings_initializer=create_initializer(stddev=stddev),
            name="token_type_embedding_layer"
        )
        
        self.layer_norm = LayerNormalization(name="layer_norm")
        self.dropout = Dropout(dropout_prob)

    def build(self, input_shape):
        self.token_embedding = self.add_weight(
            name="token_embedding",
            shape=(self.vocab_size, self.embedding_size),
            initializer=create_initializer(stddev=self.stddev)
            )
    
    def call(self, input_ids, token_type_ids, training=None):
        # input_ids: [b, s]; token_type_ids: [b, s]
        seq_len = tf.shape(input_ids)[-1]
        position_ids = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.position_embedding(position_ids)

        token_type_embeddings = self.token_type_embedding(token_type_ids)
        
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        
        return (embeddings, self.token_embedding)

    def get_config(self):
        config = super(EmbeddingLayer, self).get_config()
        config.update({"vocab_size": self.vocab_size,
                       "max_position_length": self.max_position_length,
                       "embedding_size": self.embedding_size,
                       "type_vocab_size": self.type_vocab_size,
                       "dropout_prob": self.dropout_prob,
                       "stddev": self.stddev})
        return config


class AttentionLayer(Layer):
    def __init__(self,
                hidden_size,
                num_attention_heads=1,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                query_act=None,
                key_act=None,
                value_act=None,
                name="attention_layer",
                **kwargs):
        super(AttentionLayer, self).__init__()

        assert hidden_size % num_attention_heads == 0, "The hidden size (%d) is not a multiple of the number of \
            attention heads (%d)" % (hidden_size, num_attention_heads)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        
        self.size_per_head = int(hidden_size / num_attention_heads)
        
        # query_layer, key_layer and value_layer, have the same inputs: [batch_size, seq_length, hidden_size],
        # also have the same outputs: [batch_size, seq_length, hidden_size]
        self.query_layer = Dense(hidden_size, 
                activation=query_act, 
                kernel_initializer=create_initializer(initializer_range),
                name="query_layer")
        self.key_layer = Dense(hidden_size, 
                activation=key_act, 
                kernel_initializer=create_initializer(initializer_range),
                name="key_layer")
        self.value_layer = Dense(hidden_size, 
                activation=value_act,
                kernel_initializer=create_initializer(initializer_range),
                name="value_layer")
        
        self.attn_softmax_layer = Softmax()
        self.attn_dropout_layer = Dropout(self.attention_probs_dropout_prob)
    
    def process_attention_mask(self, attention_mask):
        # attention_mask = [batch_size, seq_length]
        # assert len(tf.shape(attention_mask)) == 2, "rank of attention mask must equal to 2, [batch_size, seq_length]"
        batch_size = tf.shape(attention_mask)[0]
        seq_length = tf.shape(attention_mask)[1]

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
    
    def call(self, inputs, attention_mask=None, training=None):
        # inputs is the output of embedding layer or the output of previous encoder block
        # inputs = [batch_size, seq_length, hidden_size]
        # assert len(tf.shape(inputs)) == 3, "rank of input_ids must equal to 3, [batch_size, seq_length, hidden_size]"
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

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
        attention_probs = self.attn_softmax_layer(attention_scores)
        attention_probs = self.attn_dropout_layer(attention_probs, training=training)

        # value = [batch_size, num_attention_heads, seq_length, size_per_head]
        value = self.transpose_for_scores(value, batch_size, self.num_attention_heads, seq_length, self.size_per_head)

        # outputs = [batch_size, num_attention_heads, seq_length, size_per_head]
        outputs = tf.matmul(attention_probs, value)
        # outputs = [batch_size, seq_length, num_attention_heads, size_per_head]
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        # outputs = [batch_size, seq_length, hidden_size]
        outputs = tf.reshape(outputs, [batch_size, seq_length, self.hidden_size])

        return outputs
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"hidden_size": self.hidden_size,
                       "num_attention_heads": self.num_attention_heads,
                       "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                       "initializer_range": self.initializer_range,
                       "query_act": self.query_act,
                       "key_act": self.key_act,
                       "value_act": self.value_act,
                       "name": self.name
                    })
        return config


class Encoder(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 hidden_dropout_prob,
                 intermediate_size,
                 hidden_act,
                 initializer_range,
                 name="encoder_layer",
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        

        self.attn_layer = AttentionLayer(hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                name="attention_layer")
        self.attn_dense_layer = Dense(hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="attention_dense_layer")
        self.attn_dropout_layer = Dropout(attention_probs_dropout_prob,
                name="attention_dropout_layer")
        self.attn_layerNorm_layer = LayerNormalization(name="attention_layerNorm_layer")

        self.inter_encoder_layer = Dense(intermediate_size,
                activation=hidden_act,
                kernel_initializer=create_initializer(initializer_range),
                name="intermediate_encoder_layer")
        self.inter_decoder_layer = Dense(hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="intermediate_decoder_layer")
        self.inter_dropout_layer = Dropout(hidden_dropout_prob,
                name="intermediate_dropout_layer")
        self.inter_layerNorm_layer = LayerNormalization(name="intermediate_layerNorm_layer")

    def call(self, inputs, input_mask, training=None):
        attn_output = self.attn_layer(inputs, input_mask)
        attn_output = self.attn_dense_layer(attn_output)
        attn_output = self.attn_dropout_layer(attn_output, training=training)
        attn_output = self.attn_layerNorm_layer(attn_output + inputs)

        inter_output = self.inter_encoder_layer(attn_output)
        outputs = self.inter_decoder_layer(inter_output)
        outputs = self.inter_dropout_layer(outputs, training=training)
        outputs = self.inter_layerNorm_layer(outputs + attn_output)

        return outputs
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"hidden_size": self.hidden_size,
                       "num_attention_heads": self.num_attention_heads,
                       "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                       "hidden_dropout_prob": self.hidden_dropout_prob,
                       "intermediate_size": self.intermediate_size,
                       "hidden_act": self.hidden_act,
                       "initializer_range": self.initializer_range
                    })
        return config


class BertModel(Model):
    def __init__(self,
                 config:BertConfig,
                 with_nsp=False,
                 with_mlm=False,
                 is_pretrain=False,
                 name="bert_model",
                 **kwargs):
        super(BertModel, self).__init__(**kwargs)

        self.config = config
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.is_pretrain = is_pretrain

        self.embedding_layer = EmbeddingLayer(
                vocab_size=config.vocab_size,
                max_position_length=config.max_position_embeddings,
                embedding_size=config.hidden_size,
                type_vocab_size=config.type_vocab_size,
                dropout_prob=config.hidden_dropout_prob,
                stddev=config.initializer_range)
        
        self.encoder_layers = [
            Encoder(hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    initializer_range=config.initializer_range)
                    for _ in range(config.num_hidden_layers)
        ]

        self.pooled_dense_layer = Dense(config.hidden_size,
                activation="tanh",
                kernel_initializer=create_initializer(config.initializer_range),
                name="pooled_dense_layer")
        
        if self.with_nsp:
            self.nsp_dense_layer = Dense(2,
                kernel_initializer=create_initializer(config.initializer_range),
                name="nsp_dense_layer")
            self.nsp_softmax_layer = Softmax(name="nsp_softmax_layer")
        
        if self.with_mlm:
            self.mlm_dense_layer = Dense(config.hidden_size,
                    activation=config.hidden_act,
                    kernel_initializer=create_initializer(config.initializer_range),
                    name="mlm_dense_layer")
            self.mlm_layerNorm_layer = LayerNormalization(name="mlm_layerNorm_layer")
            self.mlm_softmax_layer = Softmax(name="mlm_softmax_layer")
    
    def build(self, input_shape):
        self.mlm_bias = self.add_weight(
            name="mlm_bias",
            shape=(self.config.vocab_size,),
            initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, training=None):
        # if with_mlm=True, inputs will be (input_ids, input_mask, token_type_ids, masked_lm_positions)
        # else, inputs will be (input_ids, input_mask, token_type_ids)
        (input_ids, input_mask, token_type_ids) = inputs[:3]
        if self.with_mlm:
            masked_lm_positions = inputs[3]

        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        
        # assert check_input_mask(input_mask), "values of input_mask must be 0 or 1"
        # assert check_token_type_ids(token_type_ids, self.config.type_vocab_size), "num of token types in token_type_ids\
                # must not be larger than type_vocab_size in config (%d)" % (self.config.type_vocab_size)

        self.embedding_output, self.embedding_table = self.embedding_layer(input_ids, token_type_ids, training=training)

        self.all_encoder_outputs = []

        prev_output = self.embedding_output
            
        for layer_idx in range(self.config.num_hidden_layers):
            encoder_layer = self.encoder_layers[layer_idx]
            curr_output = encoder_layer(inputs=prev_output, input_mask=input_mask, training=training)
            prev_output = curr_output
            self.all_encoder_outputs.append(curr_output)

        # self.sequence_output = [batch_size, seq_length, hidden_size]
        self.sequence_output = self.all_encoder_outputs[-1]

        # first_token_output = [batch_size, hidden_size]
        first_token_output = self.sequence_output[:, 0]
        # self.pooled_output = [batch_size, hidden_size]
        self.pooled_output = self.pooled_dense_layer(first_token_output)

        # nsp task
        if self.with_nsp:
            # nsp_outputs = [batch_size, 2]
            nsp_outputs = self.nsp_dense_layer(self.pooled_output)
            self.nsp_outputs = self.nsp_softmax_layer(nsp_outputs)
        
        # mlm task
        if self.with_mlm:
            # mlm_outputs = [batch_size, seq_length, hidden_size]
            mlm_outputs = self.mlm_dense_layer(self.sequence_output)
            mlm_outputs = self.mlm_layerNorm_layer(mlm_outputs)
            
            # mlm_outputs = [batch_size, seq_length, vocab_size]
            mlm_outputs = tf.matmul(mlm_outputs, self.embedding_table, transpose_b=True)
            mlm_outputs += self.mlm_bias
            mlm_outputs = self.mlm_softmax_layer(mlm_outputs)
            
            # self.mlm_outputs = [batch_size, max_predications_per_seq, vocab_size]
            self.mlm_outputs = tf.gather(mlm_outputs, masked_lm_positions, axis=1, batch_dims=-1)
        
        if self.is_pretrain:
            return (self.nsp_outputs, self.mlm_outputs)
        else:
            return self.sequence_output

    def get_pooled_output(self):
        return self.pooled_output
    
    def get_sequence_output(self):
        return self.sequence_output
    
    def get_all_encoder_outputs(self):
        return self.all_encoder_outputs
    
    def get_embedding_output(self):
        return self.embedding_output
    
    def get_embedding_table(self):
        return self.embedding_table

    def get_nsp_outputs(self):
        return self.nsp_outputs
    
    def get_mlm_outputs(self):
        return self.mlm_outputs

    def get_config(self):
        config = super(BertModel, self).get_config()
        config.update({"config": self.config,
                       "with_nsp": self.with_nsp,
                       "with_mlm": self.with_mlm,
                       "is_pretrain": self.is_pretrain,
                       "get_pooled_output": self.get_pooled_output(),
                       "get_sequence_output": self.get_sequence_output(),
                       "get_all_encoder_outputs": self.get_all_encoder_outputs(),
                       "get_embedding_output": self.get_embedding_output(),
                       "get_embedding_table": self.get_embedding_table()
                       })
        return config


# def mlm_loss_wrapper(inputs):
#     def mlm_loss(y_true, y_pred):
#         # y_true = [batch_size, MAX_PREDICTIONS_PER_SEQ]
#         # y_pred = [batch_size, MAX_PREDICTIONS_PER_SEQ, vocab_size]
#         # masked_lm_weigths = [batch_size, MAX_PREDICTIONS_PER_SEQ]
#         masked_lm_weights = inputs[-1]

#         scc = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

#         # mlm_loss = [batch_size, MAX_PREDICTIONS_PER_SEQ]
#         mlm_loss = scc(y_true, y_pred)
#         numerator = tf.reduce_sum(masked_lm_weights * mlm_loss)
#         denominator = tf.reduce_sum(masked_lm_weights) + 1e-5
#         mlm_loss = numerator / denominator
#         return mlm_loss
#     return mlm_loss

# custom loss part
def mlm_loss(y_true, y_pred):
    mlm_weights = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    scc = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    mlm_loss = scc(y_true, y_pred)
    numerator = tf.reduce_sum(mlm_loss * mlm_weights)
    denominator = tf.reduce_sum(mlm_weights) + 1e-5
    mlm_loss = numerator / denominator
    return mlm_loss

# custom metrics part
def nsp_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)
    metric = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
    numerator = tf.reduce_sum(metric)
    denominator = tf.cast(tf.shape(y_true)[-1], dtype=tf.float32)
    metric = numerator / denominator
    return metric

def mlm_accuracy(y_true, y_pred):
    # y_true = [batch_size, MAX_PREDICTIONS_PER_SEQ]
    # y_pred = [batch_size, MAX_PREDICTIONS_PER_SEQ, vocab_size]
    y_true = tf.cast(y_true, dtype=tf.float32)
    mlm_weights = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    # y_pred = [batch_size, MAX_PREDICTIONS_PER_SEQ]
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)
    y_pred = y_pred * mlm_weights
    metric = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
    numerator = tf.reduce_sum(metric * mlm_weights)
    denominator = tf.reduce_sum(mlm_weights) + 1e-5
    metric = numerator / denominator
    return metric
