import numpy as np
import librosa
import os

def load_audio_file(file_path):
    # 加载音频文件，并转换采样率
    wave, sr = librosa.load(file_path, sr=16000, mono=True)
    return wave

def extract_features(audio, yamnet):
    # YAMNet 返回三个输出：scores, embeddings, spectrogram
    scores, embeddings, spectrogram = yamnet(audio)
    # 根据需要截断或填充 embeddings 以适应模型输入
    max_time_steps = 60  # 假设最大时间步长
    if embeddings.shape[0] < max_time_steps:
        # 填充
        embeddings = np.pad(embeddings, ((0, max_time_steps - embeddings.shape[0]), (0, 0)), 'constant')
    elif embeddings.shape[0] > max_time_steps:
        # 截断
        embeddings = embeddings[:max_time_steps]
    return embeddings

def process_folder(folder_path,yamnet):
    # 创建一个字典来存储每个文件的 embeddings
    all_embeddings = {}
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp3'):  # 确保处理 MP3 文件
            file_path = os.path.join(folder_path, filename)
            audio = load_audio_file(file_path)
            embeddings = extract_features(audio, yamnet)
            all_embeddings[filename] = embeddings # 将 embeddings 转换为 NumPy 数组，并存储
    return all_embeddings
