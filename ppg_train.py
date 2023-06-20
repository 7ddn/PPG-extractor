''' Use a simple CNN for test, should use something like transformer for further research. '''

import tensorflow as tf
import os
from tqdm import tqdm
import pickle

database_dir = os.path.expanduser('~/l2_arctic/dataset')

ds = tf.data.Dataset.load(database_dir)

def remove_letter(text):
    return ''.join(i for i in text if not i.isdigit())

tv_dict_path = "tv_layer.pkl"
# TODO: save and load tv_layer
# TODO: move path to hyperparameter file


token_layer = tf.keras.layers.TextVectorization(standardize = remove_letter)
token_layer.adapt(ds.map(lambda wav, phone, speaker, filename: phone)) 

phone_table = token_layer.get_vocabulary()
