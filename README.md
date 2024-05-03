# Music Recommendation System Based on Color-Based Emotions
[KaiyangZhao](https://github.com/KaiyangZhao0603) [Duanning Wang](https://github.com/119010291) [Zhinuo Li](https://github.com/zhinuo5375)

Welcome to our Color Emotion Music Recommender! This system helps you find music that matches how you're feeling just by picking a color. Here’s how you can use it:

1. **Pick a Color**: Choose a color that shows how you feel right now.
2. **We Understand Your Feeling**: Our system uses the color you pick to figure out your mood.
3. **Get Music That Fits Your Mood**: After understanding your mood, we give you a playlist of songs from any music library that goes well with how you feel.



## General flowchart
![alt text](flowchart.16.png?raw=true)
## Datasets
training dataset: DEAM  
https://cvml.unige.ch/databases/DEAM/  
* Comprehensive genres
* Annotated with emotion labels(Arousal & Valence )
* Provides audio files with the same length (30 seconds)

recommendation dataset: fma  
https://github.com/mdeff/fma
* Large dataset scale
* Comprehensive metadate

## Data preprocess
#### 1. load and match data
* note here that when load mp3. music, if start_time=0, it will get error: Input signal length=0 is too small to resample from. You can cut the audio to fix it (what we did) or other methods according to https://blog.csdn.net/sxf1061700625/article/details/128950827.
* not all audio have valence and arousal, so we need to align them according to the song_ids first.
#### 2. data augmentation (5406 songs in total)
We tried augmentation methods including pitch shifting, add background noise, reverb and many kinds of filters. Finally, we choose pitch shifting and a lowpass filter (ladder filter), but we put the code of all of augmentation methods in the utils.py.
#### 3. data split
20% for test (1082)  
80% for training  
｜- 25% for validation (1081)  
｜- 75% for training (3243)  

## Model
we use yamnet to extract features and apply a TCN model after that.

## Using this model
## Reference
Dharmapriya, J., Dayarathne, L., Diasena, T., Arunathilake, S., Kodikara, N., & Wijesekera, P. (2021, January). Music Emotion Visualization through Colour. In 2021 International Conference on Electronics, Information, and Communication (ICEIC) (pp. 1-6). IEEE.
