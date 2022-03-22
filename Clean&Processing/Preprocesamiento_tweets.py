# preprocesamiento
import spacy
import nltk
import string
import numpy as np
from collections import Counter
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

stemmer = SnowballStemmer('english')

# preprocesamiento
def clean_tweet(text, stopw = True, idiom = 'english', stem = False ):
    # elimina usuarios y URLS
    text = ' '.join(re.sub("(#\S+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    # elimina múltiples espacios en blanco
    text = re.sub(r' +', ' ', text)
    # elimina el salto de línea
    text = re.sub(r'\n', ' ', text)
    # Elimina palabras menores a 2 letras
    #for token in text.split():
      #Si deseamos quitar stopwords
    if stopw == True:
      text = ''.join([token+' ' for token in  text.split() if (token not in stopwords.words(idiom)) & (len(token)> 2) & (token.isalpha())])
    #Si queremos hacer stemming
    if stem == True:
      stemmer = SnowballStemmer(idiom)
      text = ''.join([stemmer.stem(token)+' ' for token in  text.split()])
    return text.lower()

def clean(data, column_text, stopw = True, idiom = 'english', stem = False ):
  clean_df = data.copy()
  clean_df[column_text] = data.apply(lambda row : clean_tweet(row[column_text]), axis=1)
  #Eliminamos vacios y duplicados
  clean_df = clean_df.dropna().drop_duplicates()
  for index, row in clean_df.iterrows():
    if row[column_text].strip() == "":
      clean_df.drop([index], inplace = True )
    # eliminar tweets menores a 5 palabras
    elif len(row[column_text].split()) < 5: 
      clean_df.drop([index], inplace = True )
  return clean_df