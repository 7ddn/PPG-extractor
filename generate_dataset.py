import tensorflow as tf
import os
from textgrids import TextGrid
import librosa
from tqdm import tqdm


arctic_dir = os.path.expanduser('~/l2arctic')
sr = 16000 # downsample from 44100
frame_length = 80 # not sure, to be determined 
frame_time = frame_length / sr
eps = 1e-5

wavs = []
phones = []
speakers = []
files = []

data = []

for speaker in os.listdir(arctic_dir):
    
    if speaker == 'dataset':
        continue
    print('Speaker ', speaker)
     
    wav_path = os.path.join(arctic_dir, speaker, 'wav')
    textgrid_path = os.path.join(arctic_dir, speaker, 'textgrid')
    # wavs = []
    # phones = []
    # speakers = []
    # files = []
    for phone_file in tqdm(os.listdir(textgrid_path)):
        #filename.append(phone_file)
        fl = phone_file
        if phone_file.startswith('.'):
            continue
        wav_file = os.path.join(wav_path, phone_file.replace('TextGrid', 'wav'))
        phone_file = os.path.join(textgrid_path, phone_file)
        wav, _ = librosa.load(path = wav_file, sr = sr)
        phone = []
        tg = TextGrid()
        tg.read(phone_file)
        phs = tg['phones']
    
        #if '0043' in phone_file:
            #print(phs)

        assert(round(phs[-1].xmin / frame_time) * frame_time - phs[-1].xmin < eps)
        for ph in phs:
            if ph.text == '':
                break
            phone = phone + [ph.text] * round(ph.dur / frame_time)
        phone = tf.convert_to_tensor(phone, dtype = tf.string)

        
        phones.append(phone)
        wavs.append(wav)
        speakers.append(speaker)
        files.append(fl)

        data.append((tf.convert_to_tensor(wav), phone, tf.convert_to_tensor(speaker), tf.convert_to_tensor(fl)))
    # sp = [speaker] * len(wavs)
    # speakers = speakers + sp    

wavs = tf.ragged.stack(wavs, axis=0)
phones = tf.ragged.stack(phones, axis=0)
speakers = tf.stack(speakers, axis=0)
files = tf.stack(files, axis = 0)

ds = tf.data.Dataset.from_tensor_slices((wavs, phones, speakers, files))

# ds = tf.data.Dataset.from_tensor_slices(data)
path = os.path.join(arctic_dir, 'dataset')
ds.save(path)
    
    
