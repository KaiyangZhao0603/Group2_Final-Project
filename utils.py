import os
import csv
import numpy as np
import random
import tensorflow as tf
import keras
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Load data from the original music dataset and return the audio file paths
# and their corresponding labels according to labels' id.
# both outputs are lists
def load_data(audio_data_home, label_list, label_ids):

    labels = []
    audio_file_paths = []

    for id in label_ids:

        # store [arousal, valence] label pairs into labels
        labels.append(label_list[id])

        # put corresponding audio path into audio_file_paths
        audio_file_paths.append(os.path.join(audio_data_home, id, '.mp3'))
    
    return audio_file_paths, labels


# Generator function that yields audio and labels from the specified dataset,
# with optional data augmentation.
def wav_generator(data_home, augment, label_list, label_ids, sample_rate=22050,
                  shuffle=True):
    
    audio_file_paths, labels = load_data(data_home, label_list, label_ids)

    # Convert labels to numpy array
    labels = np.array(labels)

    # Shuffle data
    if shuffle:
        idxs = np.random.permutation(len(labels))
        audio_file_paths = [audio_file_paths[i] for i in idxs]
        labels = labels[idxs]

    for idx in range(len(audio_file_paths)):

        # Load audio at given sample_rate and label
        label = labels[idx]
        audio, _ = librosa.load(audio_file_paths[idx], sr=sample_rate)

        # Shorten audio to 29s due to imprecisions in duration of GTZAN
        # (ensures same duration files)
        audio = audio[:29*sample_rate]
        
        # Apply augmentation
        if augment:
            pass
        yield audio, label


def create_dataset(data_generator, input_args, input_shape):

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=input_args,
        output_signature=(
          tf.TensorSpec(shape=input_shape, dtype=tf.float32),
          tf.TensorSpec(shape=(), dtype=tf.uint8)
          )
        )

    return dataset


# Run YAMNet to extract embeddings from the wav data. 
def extract_yamnet_embedding(wav_data, yamnet):
    scores, embeddings, spectrogram = yamnet(wav_data)
    return embeddings