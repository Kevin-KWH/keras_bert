import tensorflow as tf

from model import BertConfig, EmbeddingLayer, AttentionLayer, Encoder, BertModel
from model import mlm_loss, nsp_accuracy, mlm_accuracy

config = BertConfig(vocab_size=300,
                    hidden_size=128,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    intermediate_size=512,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=512,
                    type_vocab_size=2,
                    initializer_range=0.02)


bert_model = BertModel(config, with_nsp=True, with_mlm=True, is_pretrain=True)



# if with_mlm=True, inputs will be (input_ids, input_mask, token_type_ids, masked_lm_positions)
input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32)
input_mask = tf.keras.Input(shape=(512,), dtype=tf.int32)
token_type_ids = tf.keras.Input(shape=(512,), dtype=tf.int32)
# masked_lm_positions = [batch_size, MAX_PREDICTIONS_PER_SEQ]
masked_lm_positions = tf.keras.Input(shape=(2,), dtype=tf.int32)

inputs = (input_ids, input_mask, token_type_ids, masked_lm_positions)
bert_model(inputs)

# bert_model = tf.keras.Model(inputs, outputs)

bert_model.compile(optimizer="adam",
                   loss={"output_1": tf.keras.losses.SparseCategoricalCrossentropy(),
                         "output_2": mlm_loss},
                   loss_weights={"output_1": 1, "output_2": 1},
                   metrics={"output_1": nsp_accuracy, "output_2": mlm_accuracy}
                )

bert_model.summary(line_length=130)

# inputs
train_input_ids = tf.random.uniform(shape=[30, 111], maxval=10, dtype=tf.int32)
train_input_mask = tf.random.uniform(shape=[30, 111], maxval=2, dtype=tf.int32)
train_token_type_ids = tf.zeros(shape=[30, 111], dtype=tf.int32)
train_masked_lm_positions = tf.random.uniform(shape=[30, 2], maxval=111, dtype=tf.int32)

train_inputs = (train_input_ids, train_input_mask, train_token_type_ids, train_masked_lm_positions)

# outputs
train_next_sentence_labels = tf.random.uniform(shape=[30], maxval=2, dtype=tf.int32)
train_mlm_label_ids = tf.random.uniform(shape=[30, 2], maxval=300, dtype=tf.int32)

train_outputs = (train_next_sentence_labels, train_mlm_label_ids)

bert_model.fit(x=train_inputs, y=train_outputs, epochs=10)
# print(bert_model.metrics_names)

bert_model.save("saved_model/bert")
