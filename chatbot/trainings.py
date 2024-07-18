import json
import pickle
import numpy as np
import random
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

# k=nltk.download('wordnet')
read=open("intents.json").read()
# json.loads("intents.json")
intents=json.loads(read)
# print("TYPE:",type(intents))
words=[];classes=[];documents=[]
ignore_letters=["?",".","!","'",","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if(intent['tag'] not in classes):
            classes.append(intent['tag'])
# print(word_list)
# print(words)
# print(documents)
lemmatize=WordNetLemmatizer()
word=[lemmatize.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))
# print(words)
classes=sorted(set(classes))

# print(classes)
# cls=pickle.dump(words,wordpickle)
# wrd=pickle.dump(classes,classpickle)
# classpickle=open('words.pkl','wb')
# cls=pickle.dump(words,wordpickle)
# word=pickle.dump(classes,classpickle)

wordspickle=open('words.pkl','wb')
classespickle=open('classes.pkl','wb')
wrd=pickle.dump(words,open('words.pkl','wb'))
cls=pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

for document in documents:
    bag=[]
    word_pattern=document[0]
    word_pattern=[lemmatize.lemmatize(word.lower())
                  for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
    output_raw=list(output_empty)
    output_raw[classes.index(document[1])]=1
    training.append([bag,output_raw])

random.shuffle(training)
training=np.array(training)

# print(training)

train_x=list(training[:,0])
train_y=list(training[:,1])

# print(train_x)
# print(training)

#create the model 
model=Sequential()
model.add(Dense(128,activation='relu',input_size=len(train_x[0])))
model.add(Dropout(0.5))

model.add(Dense(64,Activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-5,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(np.array(train_x),np.array(train_y),epoch=200,batch_size=5,verbose=1)
model.save("chatbot_model.model")
print("Done")