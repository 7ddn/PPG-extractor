import tensorflow as tf
import os
import pickle
from ppg_train import remove_letter, PPG_CNN
import pandas as pd
import seaborns as sbs
import numpy as np

tv_dict_path = "tv_layer.pkl"

if os.path.isfile(tv_dict_path):
    from_disk = pickle.load(open(tv_dict_path, "rb"))
    token_layer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    token_layer.set_weights(from_disk['weights'])

dict = token_layer.get_vocabulary()

model = PPG_CNN(tokenizer = token_layer)

cp_dir = './checkpoints'
latest = tf.train.latest_checkpoint(cp_dir)
model.load_weights(latest)

db_d = os.path.expanuser('~/l2arctic/dataset')
ds = tf.data.Dataset.load(db_d)

for sample in ds.take(1):
    mel, token = sample

pred = model(mel)

ph_pred = [dict[i.numpy()] for i in tf.argmax(pred, axis=-1)]

print(ph_pred) 
