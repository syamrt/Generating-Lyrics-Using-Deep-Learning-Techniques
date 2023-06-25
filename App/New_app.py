import streamlit as st
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import string, os
import nltk
import re
# import keras
import random
import io
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adamax
import sys
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
# Image = Image.open('Lyrics3.jpg')
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Lyrics_generator",page_icon=":notes:", layout="wide")

# ## loading data

data = pd.read_csv("/Users/giridharana.r/Desktop/DP_1/Songs.csv")
### ----------
#Lining up all the lyrics to create corpus
Corpus =''
for listitem in data.Lyrics:
    Corpus += listitem
    
Corpus = Corpus.lower() #converting all alphabets to lowecase 

# ### ----------
# #Keeping only a limited set of characters. 
to_remove = ['{', '}', '~', '©', 'à', 'á', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'í', 'ñ', 'ó', 'ö', 'ü', 'ŏ',
             'е', 'ا', 'س', 'ل', 'م', 'و', '\u2005', '\u200a', '\u200b', '–', '—', '‘', '’', '‚', '“', '”', 
             '…', '\u205f', '\ufeff', '!', '&', '(', ')', '*', '-',  '/', ]
for symbol in to_remove:
    Corpus = Corpus.replace(symbol," ")

# ### ------------
# # Storing all the unique characters present in my corpus to build a mapping dic. 
symb = sorted(list(set(Corpus)))

L_corpus = len(Corpus) #length of corpus
L_symb = len(symb) #length of total unique characters

#Building dictionary to access the vocabulary from indices and vice versa
mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

#---------------
#Splitting the Corpus in equal length of strings and output target
length = 40
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])

L_datapoints = len(targets)
# # print("Total number of sequences in the Corpus:", L_datapoints)

#--------------- Loading models 1----------------------------------------

# model_1=tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/Models/my_gru_model_1.h5')
# model_2=tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/Models/Double_LSTM_updated_1.h5')
# model_3=tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/Models/Bidirectional_model.h5')
# @st.cache(allow_output_mutation=True)
# def load_model1():

  # return model1
# with st.spinner('Model is being loaded..'):
#   model1=load_model1()
#--------------- Loading models 2----------------------------------------

# def load_model2():

#   return model2
# with st.spinner('Model is being loaded..'):
#   model2=load_model2()

#--------------- Loading models 3----------------------------------------
# def load_model3():

#   return model3
# with st.spinner('Model is being loaded..'):
#   model3=load_model3()
st.markdown("<h1 style='text-align: center; color: White'>Lyrics Generator Application</h1>", unsafe_allow_html=True)
# st.image(image=Image)
selected_model = st.selectbox("Select the model to be implemented",options = ["GRU","Double_LSTM","Bidirectional_LSTM","Bidirectional_LSTM_GRU"])
# @st.cache
if selected_model == "DOUBLE_LSTM":
  model = tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/New app/Final_LSTM_2_model.h5')
if selected_model == "Double_GRU":
  model = tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/New app/')
if selected_model == "Bidirectional_LSTM_GRU":
  model = tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/Models/Bidirectional_LSTM_GRU_model_updated.h5')
else:
  model = tf.keras.models.load_model('/Users/giridharana.r/Desktop/DP_1/Models/Bidirectional_model.h5')

st.write("The selected model is",selected_model)

#--------------- Creating the lyrics generator----------------------------------------

input = st.text_input(label="Write the lyrics",value ="Whats up baby doll how are you doing, I ",placeholder="Write the lyrics")
input = input.lower()
char_count = int(st.text_input(label="Character_count",value = "100",placeholder="Number of characters to be present"))
# st.write("The selected model is",type(model))

#--------------- Lyrics Generator 1----------------------------------------

def Lyrics_Generator(starter,Ch_count): #,temperature=1.0):
    generated= ""
    starter = starter 
    seed=[mapping[char] for char in starter]
    generated += starter 
    # Generating new text of given length
    for i in range(Ch_count):
        seed=[mapping[char] for char in starter]
        x_pred = np.reshape(seed, (1, len(seed), 1))
        x_pred = x_pred/ float(L_symb)
        prediction = model.predict(x_pred, verbose=0)[0]  
        # Getting the index of the next most probable index
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / 1.0 
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, prediction, 1)
        index = np.argmax(prediction)
        next_char = reverse_mapping[index]  
        # Generating new text
        generated += next_char
        starter = starter[1:] + next_char
       
    return generated

#--------------- Lyrics Generator 2----------------------------------------
st.write("Generated Lyrics are:")
st.write(Lyrics_Generator(starter=input,Ch_count = char_count))
#--------------- Lyrics Generator 2----------------------------------------
# new = print(Lyrics_Generator(starter=input,Ch_count=100))

# st.button(label="Load Model",on_click=load_model)
# st.button(label="Generate",on_click=Lyrics_Generator(input,100))
## -----
# # #Generating a song from the model
# song_1 = Lyrics_Generator(input,100)
#Let's have a look at the song
# st.write(Lyrics_Generator(input,100))



### Saved prompt
### Whats up baby doll how are you doing,I a


#### Bi directional LSTM

### the shoe shrunk, and the school belt got

# GRU output
# whats up baby doll how are you doing i am in the summer of the street i wanna be your end on my baby i know i need you in the same of your life i can't take it all the way i know i can't help you i know i'm gonna be a long, long time i want

