import numpy as np
import librosa
import os
from scipy import signal
from audiomentations import Compose, HighShelfFilter, LowShelfFilter, AddGaussianNoise
from pedalboard import *
import matplotlib.pyplot as plt
import pygame
from ipywidgets import Button, Output, HBox, VBox


def load_audio_file(file_path):
    # Load the audio file and convert the sampling rate
    wave, sr = librosa.load(file_path, sr=16000, duration=44, mono=True)
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


def process_folder(folder_path, yamnet):
    # Create a dictionary to store the embeddings of each file
    all_embeddings = {}
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp3'):  # Ensure to process MP3 files
            file_path = os.path.join(folder_path, filename)
            audio_original = load_audio_file(file_path)
            audio0 = pitch_shift(audio_original)    # augmentation
            # audio1 = add_bgn(audio_original)
            # audio2 = reverb(audio_original)
            audio3 = lowpass(audio_original)
            # audio_f = lowshelf(audio_original)

            # use yamnet to extract features
            embeddings_ori = extract_features(audio_original, yamnet)
            embeddings0 = extract_features(audio0, yamnet)
            # embeddings1 = extract_features(audio1)
            # embeddings2 = extract_features(audio2)
            embeddings3 = extract_features(audio3, yamnet)
            # embeddings_f = extract_features(audio_f)

            all_embeddings[filename] = embeddings_ori
            all_embeddings[filename + '0aug'] = embeddings0
            # all_embeddings[filename + '1aug'] = embeddings1
            # all_embeddings[filename + '2aug'] = embeddings2
            all_embeddings[filename + '3aug'] = embeddings3
            # all_embeddings[filename + 'f'] = embeddings_f

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


# Plot the training and validation loss and mae.
def plot_loss(history):

    mae = history.history["mae"]
    val_mae = history.history["val_mae"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(mae) + 1)
    plt.plot(epochs, mae, "bo", label="Training mae")
    plt.plot(epochs, val_mae, "b", label="Validation mae")
    plt.title("Training and validation mae")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


# Function to calculate Manhattan distance
def calculate_manhattan_distance(x, y, valence, arousal):
    return abs(x - valence) + abs(y - arousal)


def play_audio(file_path):
    # load
    pygame.mixer.music.load(file_path)
    
    # play
    pygame.mixer.music.play()


def pause_audio():
    pygame.mixer.music.pause()


def unpause_audio():
    pygame.mixer.music.unpause()


def find_audio_files(base_path, file_ids):
    audio_files = []
    
    for root, dirs, files in os.walk(base_path):
        # check if the files are in file_ids 
        for file in files:
            # get the number part of filename
            file_id = os.path.splitext(file)[0]
            try:
                if float(file_id) in file_ids:
                    file_path = os.path.join(root, file)
                    audio_files.append(file_path)
            except ValueError:
                continue
    
    return audio_files


def create_audio_controls(track_id, file_path):
    play_button = Button(description=f"Play: {track_id}")
    pause_button = Button(description="Stop")
    output = Output()
    
    def on_play_button_clicked(b):
        with output:
            print(f"Playing: {track_id}")
            play_audio(file_path)
    
    def on_pause_button_clicked(b):
        if pygame.mixer.music.get_busy():
            if pygame.mixer.music.get_pos() > 0:
                with output:
                    print("Pausing audio")
                pause_audio()
            else:
                with output:
                    print("Resuming audio")
                unpause_audio()
    
    play_button.on_click(on_play_button_clicked)
    pause_button.on_click(on_pause_button_clicked)
    
    return VBox([HBox([play_button, pause_button]), output])


# ================================= following are different kinds of augmentation methods ======================================
def pitch_shift(audio, sample_rate=22050, pitch_shift_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift_steps)


def add_bgn(audio, sample_rate=22050):
    noise = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)])
    return noise(audio, sample_rate)


def reverb(audio):
    board = Pedalboard([Chorus(), Reverb(room_size=0.25)])
    return board(audio, 22050)


# ladder filter
def lowpass(audio):
    board = Pedalboard([Chorus(),
                    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900)])
    return board(audio, 22050)


# filters for augmentation
def butterworth(audio):
    b, a = signal.butter(4, 1/16, 'low', analog=False)
    return signal.filtfilt(b, a, audio)


def chev1(audio):
    b, a = signal.cheby1(4, 5, 1/16, 'low', analog=False)
    return signal.filtfilt(b, a, audio)


def chev2(audio):
    b, a = signal.cheby2(4, 50, 1/16, 'low', analog=False)
    return signal.filtfilt(b, a, audio)


def ellip(audio):
    b, a = signal.ellip(4, 5, 50, 1/16, 'low', analog=False)
    return signal.filtfilt(b, a, audio)


def ellipH(audio):
    b, a = signal.ellip(4, 5, 50, 1/16, 'high', analog=False)
    return signal.filtfilt(b, a, audio)


def ellipB(audio):
    b, a = signal.ellip(4, 5, 50, [1/32, 1/16], 'band', analog=False)
    return signal.filtfilt(b, a, audio)


def peak(audio):
    b, a = signal.iirpeak(200, 30, 16000)
    return signal.filtfilt(b, a, audio)


def highshelf(audio):
    noise = Compose([HighShelfFilter(min_center_freq=100, max_center_freq=1000, p=0.5)])
    return noise(audio, 16000)


def lowshelf(audio):
    noise = Compose([LowShelfFilter(min_center_freq=100, max_center_freq=1000, p=0.5)])
    return noise(audio, 16000)
