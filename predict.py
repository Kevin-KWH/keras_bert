import tensorflow as tf

from model import BertConfig, EmbeddingLayer, AttentionLayer, Encoder, BertModel
from model import mlm_loss, nsp_accuracy, mlm_accuracy

custom_objects = {
    "BertConfig": BertConfig,
    'EmbeddingLayer': EmbeddingLayer,
    'AttentionLayer': AttentionLayer,
    'Encoder': Encoder,
    'BertModel': BertModel,
    "mlm_loss": mlm_loss,
    "nsp_accuracy": nsp_accuracy,
    "mlm_accuracy": mlm_accuracy
}

model = tf.keras.models.load_model("saved_model/bert", custom_objects=custom_objects)

model.summary(line_length=130)

input_ids = tf.random.uniform(shape=[30, 512], maxval=10, dtype=tf.int32)
input_mask = tf.random.uniform(shape=[30, 512], maxval=2, dtype=tf.int32)
token_type_ids = tf.zeros(shape=[30, 512], dtype=tf.int32)
masked_lm_positions = tf.random.uniform(shape=[30, 2], maxval=512, dtype=tf.int32)

inputs = (input_ids, input_mask, token_type_ids, masked_lm_positions)

outputs = model.predict(inputs)
print(outputs)
