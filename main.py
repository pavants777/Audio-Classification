import resampy 
import pandas as pd
import os
import librosa
import librosa.display 

metadata = pd.read_csv('data.csv')
audio_data_set = 'donateacry_corpus_cleaned_and_updated_data'

def feature_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_feature = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_feature

import numpy as np
from tqdm import tqdm

extracted_feature = []

for index_num,row in tqdm(metadata.iterrows()):
    file_name = row['data']
    class_label = row['class']
    data = feature_extractor(file_name)
    extracted_feature.append([data,class_label])

extracted_features_df = pd.DataFrame(extracted_feature,columns=['feature','class'])

x = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels = y.shape[1]

model = Sequential()

model.add(Dense(100, input_shape=(40,)))  
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))  
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))  
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels)) 
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

from tensorflow.keras.callbacks import ModelCheckpoint

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_model/audio_classification.keras', verbose=1, save_best_only=True)

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer])  
test_accuracy = model.evaluate(x_test,y_test,verbose=0)


filename = r'C:\Users\pavan\Desktop\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data\discomfort\10A40438-09AA-4A21-83B4-8119F03F7A11-1430925142-1.0-f-26-dc.wav'
prediction_feature = feature_extractor(filename)

prediction_feature = prediction_feature.reshape(1, -1)
y_predict = np.argmax(model.predict(prediction_feature), axis=1)

predict_class = labelencoder.inverse_transform(y_predict)[0]
predict_class
