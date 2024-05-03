# Music Recommendation System Based on Color-Based Emotions
[KaiyangZhao](https://github.com/KaiyangZhao0603) [Duanning Wang](https://github.com/119010291) [Zhinuo Li](https://github.com/zhinuo5375)

Welcome to our Color Emotion Music Recommender! This system helps you find music that matches how you're feeling just by picking a color.

1. **Pick a Color**: Choose a color that shows how you feel right now.
2. **We Understand Your Feeling**: Our system uses the color you pick to figure out your mood.
3. **Get Music That Fits Your Mood**: After understanding your mood, we give you a playlist of songs from any music library that goes well with how you feel.

## Our Outcomes
1. We trained a predictive TCN model that takes YAMNet embeddings as input, with the DEAM dataset. It can predict song clips' emotion labels (valence and arousal).
2. We predicted the emotion labels of 7997 song clips from the fma_small dataset using our model.
3. We built a recommendation system that allows users to select a color and recommend a song list that contains 10 song clips from the fma_small dataset.
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
#### 1. Load and match data
* note here that when loading mp3. music, if start_time=0, it will get an error: Input signal length=0 is too small to resample from. You can cut the audio to fix it (what we did) or other methods according to https://blog.csdn.net/sxf1061700625/article/details/128950827.
* not all audio clips have valence and arousal, so we need to align them according to the song_ids first.
#### 2. Data augmentation (5406 songs in total)
We tried augmentation methods including pitch shifting, adding background noise, reverb, and many kinds of filters. Finally, we chose pitch shifting and a lowpass filter (ladder filter), but we put the code of all of augmentation methods in the utils.py.
#### 3. Data split
20% for test (1082)  
80% for training  
｜- 25% for validation (1081)  
｜- 75% for training (3243)  

## Model
We use YAMNet to extract features and trained a TCN model to predict emotion labels (vanlence and arousal). 

### Model Architecture
This model is built using TensorFlow's Keras API and features a Temporal Convolutional Network (TCN) followed by a Dense layer. Below are the key components of the model:

* TCN Layer: Configured with an input shape specified by input_shape, this layer uses 64 filters and a kernel size of 6. The network includes one stack with dilations set at [1, 2, 4, 8, 16], using causal padding. Skip connections are employed to facilitate gradient flow. A dropout rate of 0.2 helps prevent overfitting, and the activation function is ReLU, with He Normal kernel initialization.
* Dense Layer: This follows the TCN layer and consists of 2 units with a linear activation function. It includes a regularization term (L2 with a factor of 0.01) to reduce overfitting.
* Batch Normalization: Applied after the Dense layer to normalize the activations of the previous layer, which helps in accelerating the training process.
* Optimizer: The model uses the Adam optimizer with a learning rate of 0.005.
Compilation: The model is compiled with the mean squared error loss function and tracks the mean absolute error (MAE) as a metric.

## How to Use Our Code
### Prepare the environment 
Ensure that your Python environment is set up correctly by installing all necessary dependencies listed in the `requirements.txt` file.

### Training Your Own Model
If you wish to train your own model, follow these steps:

1. Prepare Your Audio Data: Place your audio files in the directory `DEAM_Dataset/MEMD_audio/`. Make sure that your audio files are properly formatted and named.
   
2. Set Emotional Labels: Update the emotional labels (valence and arousal) by replacing the CSV file at `DEAM_Dataset/annotations/static_annotations_averaged_songs_1_2000.csv` with your arousal and valencce labels in cvs format.

You are welcome to use our trained model. (emotionPredict2.h5")
### Updating the Song Library
If you want to update song library, follow two steps:
1. Pleace replace your file with, `folder_path = 'fma_small/'`in the Main.ipynb, _prediction_ section
2. Generate New CSV File: After updating the song library and running the prediction script, a new CSV file will be generated with the predicted emotional labels. Replace the existing `FMA_Metadata/tracks.csv` with this new CSV to update your system's song recommendations.
### Get recommendations based on your color selection
If you want to get a recommended song list, follow these steps:

1. Download the [fma_small dataset](https://os.unil.cloud.switch.ch/fma/fma_small.zip) and [fma_metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)(you will only need the 'tracks.csv').

2. Jump to the "Recommendation" part in the Main.jpynb and replace the file path with your own in the code.

3. Run the code cells and click on the pop-up UI color palette to select a color, then you will get a song list and you will be able to listen to the corresponding audio clips from the fma_small dataset.
   
## Insights
### 1. Evaluation:
   The evaluation of music recommendation system is subjective. It is not related to the accuracy of our model, but also related to the accuracy of the emotion label provided in the training dataset. We did subjective evaluation to the song list generated by selecting different color. The result is satisfying, contracting color like blue and red will generate contrasting song in tempo, style, groove et al.. To provide more accurate objective evaluation, more participants are needed for future improvement. 
### 2. Representation of color emotion:
   In our project, emotion label for both music and color are arousal and valence. This concept comes from the article "Music Emotion Visualization through Color", which combines the color circle with 8 dimensional valence and arousal patterns. Weimport this idea in our UI part, using the relative position of the distaince from mouse click to the original to calculate the valence and arousal. In the future, if an article is published on the correspondence between RGB and arousal and valence of colors, our UI will not be limited to the color garden circle, but can be placed in any picture for color selection.
## Reference
Dharmapriya, J., Dayarathne, L., Diasena, T., Arunathilake, S., Kodikara, N., & Wijesekera, P. (2021, January). Music Emotion Visualization through Colour. In 2021 International Conference on Electronics, Information, and Communication (ICEIC) (pp. 1-6). IEEE.
