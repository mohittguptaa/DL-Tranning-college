import json 
import numpy
import pickle
import random
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
# model=load_model(open("chatbot_model.model"))

def clean_up_sentence(sentence):
    sentence_word=nltk.word_tokenize(sentence)
    sentence_word=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_word

def bag_of_words(sentence):
    sentence_word=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_word:
        for i,word in enumerate(words):
            if (word==w):
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[(i,r) for i,r in enumerate(res)
             if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for i in results:
        return_list.append({'intent':classes[i[0]],'probability':str(i[1])})
