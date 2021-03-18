import tensorflow as tf

from model import BertModel, BertConfig

model = tf.keras.models.load_model("saved_model/bert", custom_objects={"BertModel": BertModel})

model.summary()