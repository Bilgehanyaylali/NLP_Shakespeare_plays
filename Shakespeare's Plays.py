from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing
##################################################

df = pd.read_csv("shakespeare_plays.csv", sep=",")
df.head()

###############################
# Choosing play
##############################

df["play_name"].value_counts()

df = df[df["play_name"] == "Julius Caesar"]

df.head()

###############################
# Normalizing Case Folding
###############################

df['text'] = df['text'].str.lower()

###############################
# Punctuations
###############################

df['text'] = df['text'].str.replace('[^\w\s]', ' ', regex=True)

###############################
# Numbers
###############################

df['text'] = df['text'].str.replace('\d', '', regex=True)

###############################
# Stopwords
###############################

sw = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################

temp_df = pd.Series(' '.join(df['text']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

###############################
# Tokenization
###############################

df["text"].apply(lambda x: TextBlob(x).words).head()

###############################
# Lemmatization
###############################

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################################
# 2. Text Visualization
##################################################

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Wordcloud
###############################

text = " ".join(i for i in df.text)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
