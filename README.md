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
we use yamnet to extract features and apply a TCN model after that. [Download the model file here](https://example.com/path/to/emotionPredict2.h5)

## Using this model
### Prepare the environemnt 
Ensure that your Python environment is set up correctly by installing all necessary dependencies listed in the `requirements.txt` file.
This file contains our pre-trained model.[Download the model file here](https://example.com/path/to/emotionPredict2.h5)

### Training Your Own Model
If you wish to train your own model, follow these steps:

1. Prepare Your Audio Data: Place your audio files in the directory `DEAM_Dataset/MEMD_audio/`. Make sure that your audio files are properly formatted and named.
   
2. Set Emotional Labels: Update the emotional labels (valence and arousal) by modifying the CSV file at `DEAM_Dataset/annotations/static_annotations_averaged_songs_1_2000.csv` with your labels.
### Updating the Song Library
If you with to update song library, follow two steps:_
1. Pleace replace your file with, folder_path = 'fma_small/' in the training.ipynb, _prediction_ section
2. Generate New CSV File: After updating the song library and running the prediction script, a new CSV file will be generated with the predicted emotional labels. Replace the existing FMA_Metadata/tracks.csv with this new CSV to update your system's song recommendations.

## Insights
1. Evaluation:
2. Representaion of color emotion
   In our project, emotion label for both music and color are arousal and valence. This concept is from the article "Music Emotion Visualization through Colour", that combian the colour circle with 8 dimentional valence and arousal pattern. We improt this idea in out UI part, using the relative position of 
   
## Reference
Dharmapriya, J., Dayarathne, L., Diasena, T., Arunathilake, S., Kodikara, N., & Wijesekera, P. (2021, January). Music Emotion Visualization through Colour. In 2021 International Conference on Electronics, Information, and Communication (ICEIC) (pp. 1-6). IEEE.
