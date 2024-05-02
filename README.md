# Group2_Final-Project

## general flowchart
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
## Data preprocess
#### 1. load and match data
* note here that when load mp3. music, if start_time=0, it will get error: Input signal length=0 is too small to resample from. You can cut the audio to fix it (what we did) or other methods according to https://blog.csdn.net/sxf1061700625/article/details/128950827.
* not all audio have valence and arousal, so we need to align them according to the song_ids first.
#### 2. data augmentation (5406 songs in total)
We tried augmentation methods including pitch shifting, add background noise, reverb and many kinds of filters. Finally, we choose pitch shifting and a lowpass filter (ladder filter), but we put the code of all of augmentation methods in the utils.py.
#### 3. data split
20% for test (1082)  
80% for training  25% for validation (1081)  
                  75% for training (3243)  

## Model
we use yamnet to extract features and apply a TCN model after that.
