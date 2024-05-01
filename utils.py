import numpy as np
import librosa
import os

def load_audio_file(file_path):
    # Load the audio file and convert the sampling rate
    wave, sr = librosa.load(file_path, sr=16000, mono=True)
    return wave

def extract_features(audio, yamnet):
    # YAMNet returns three outputs: scores, embeddings, spectrogram
    scores, embeddings, spectrogram = yamnet(audio)
    # Truncate or pad the embeddings as needed to fit the model input
    max_time_steps = 60  # Assume maximum time steps
    if embeddings.shape[0] < max_time_steps:
        # Padding
        embeddings = np.pad(embeddings, ((0, max_time_steps - embeddings.shape[0]), (0, 0)), 'constant')
    elif embeddings.shape[0] > max_time_steps:
        # Truncating
        embeddings = embeddings[:max_time_steps]
    return embeddings

def process_folder(folder_path,yamnet):
    # Create a dictionary to store the embeddings of each file
    all_embeddings = {}
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp3'):  # Ensure to process MP3 files
            file_path = os.path.join(folder_path, filename)
            audio = load_audio_file(file_path)
            embeddings = extract_features(audio, yamnet)
            all_embeddings[filename] = embeddings # Convert embeddings to NumPy array and store
    return all_embeddings

def process_folder_FMA(folder_path, yamnet):
    all_embeddings = {}
    # Iterate through all subfolders and files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.mp3'):
                file_path = os.path.join(root, filename)
                audio = load_audio_file(file_path)
                embeddings = extract_features(audio, yamnet)
                all_embeddings[filename] = embeddings
    return all_embeddings
