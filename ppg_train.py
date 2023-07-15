''' Use a simple CNN for test, should use something like transformer for further research. '''

import tensorflow as tf
import os
from tqdm import tqdm
import pickle

database_dir = os.path.expanduser('~/l2arctic/dataset')

ds = tf.data.Dataset.load(database_dir)

def remove_letter(text):
    return tf.strings.regex_replace(text, f'[0-9]', '')

tv_dict_path = "tv_layer.pkl"
# TODO: move path to hyperparameter file

if os.path.isfile(tv_dict_path):
    from_disk = pickle.load(open(tv_dict_path, "rb"))
    token_layer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    token_layer.set_weights(from_disk['weights'])
else:
    token_layer = tf.keras.layers.TextVectorization(standardize = remove_letter)
    token_layer.adapt(ds.map(lambda wav, phone, speaker, filename: phone))
    pickle.dump({'config':token_layer.get_config(), 'weights': token_layer.get_weights()}, open(tv_dict_path, "wb"))


def get_mel(wav, sr = 16000, frame_length = 320, frame_step = 160):
    sp = tf.signal.stft(wav, frame_length = frame_length, frame_step = frame_step, pad_end = True)
    sp = tf.abs(sp)
    # print(sp.shape)
    num_sp_bins = sp.shape[-1]
    leh, ueh, num_bins = 80.0, 7600.0, 80
    mat = tf.signal.linear_to_mel_weight_matrix(
        num_bins, num_sp_bins, sr, leh, ueh)
    mel = tf.tensordot(sp, mat, 1)
    mel.set_shape(sp.shape[:-1].concatenate(mat.shape[-1:]))
    log_mel = tf.math.log(mel + 1e-6)
    # print(log_mel.shape)
    return log_mel#[:-1, :]

class PPG_CNN(tf.keras.Model):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.frame_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(64, 1, activation='relu'),
            tf.keras.layers.Conv1D(32, 1, activation='relu'),
            # tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(tokenizer.vocabulary_size()),
            tf.keras.layers.Softmax(axis = -1)])

    def call(self, spec):
        '''
        # spec shape [batch_size, num_frames + 1, fft_size]
        ppg_frames = []
        #for spec in spec_batch:
        for i in tf.range(spec.shape[0]-1):
            frame = self.frame_model(tf.slice(spec,[i, 0], [i+2, tf.shape(spec)[1]]))
            # frame shape [2, fft_size]
            ppg_frames.append(self.frame_model(frame))
            # ppg_frame shape [1, num_phoneme]
            
        # ppg = tf.concat(ppg_frames, axis = 1)
        
        return ppg
        '''
        
        spec = spec[:, tf.newaxis, :]
        return self.frame_model(spec)
            
token_ds = ds.map(lambda wav, phone, speaker, filename: (get_mel(wav), token_layer(phone)))#.map(lambda mel, token: (mel[:min(mel.shape[0], token.shape[0])], token[min(mel.shape[0], token.shape[0])]))


def split_dataset(ds, ds_size=None, train_split = 0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 10000):
    assert (train_split + test_split + val_split == 1)

    # ds =

    if ds_size is None:
        ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(buffer_size = shuffle_size, seed = 42)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_dataset(token_ds)

train_ds = train_ds#.batch(32)
val_ds = val_ds#.batch(32)
test_ds = test_ds#.batch(32)

print('Dataset Generated:\n Training Dataset Size: {train_size}\n Validation Dataset Size: {val_size}\n Test Dataset Size: {test_size}'.format(
    train_size = len(train_ds), val_size = len(val_ds), test_size = len(test_ds)))
model = PPG_CNN(tokenizer = token_layer)

def ppg_loss(y_truth, y_pred):
    # y_pred ppg shape [batch_size, num_frames, num_phoneme]
    # y_truth shape [batch_size, num_frames]
    loss = 0.    

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # for y_t, y_p in zip(y_truth, y_pred):
        # loss = loss + loss_fn(y_truth, y_pred) 
    # print(len(y_truth))
    # print(len(y_pred))
    # min_len = min(y_truth.shape[0], y_pred.shape[0])    

    loss = loss_fn(y_truth, y_pred)
    return loss

def ppg_acc(y_truth, y_pred):
    # y_pred ppg shape [batch_size, num_frames, num_phoneme]
    # y_truth shape [batch_size, num_frames]    

    p_pred = tf.cast(tf.math.argmax(y_pred), tf.int32)
    y_truth = tf.cast(y_truth, tf.int32)
    match = tf.cast(p_pred == y_truth, tf.float32)
    
    return tf.reduce_mean(match)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'sparse_categorical_accuracy')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = './checkpoints/ckp.{epoch:02d}',
    save_weights_only = True,
    save_best_only = True,
    verbose = 1)

history = model.fit(
    train_ds,
    epochs = 100,
    validation_data = val_ds,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5), checkpoint_callback])

model.evaluate(test_ds) 
