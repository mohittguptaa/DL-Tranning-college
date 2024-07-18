import numpy as np
import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

text="AIUltimators is provinding AI services and development company."
chars=sorted(list(set(text)))
# print(list(enumerate(chars)))
char_to_index={char:i for i,char in enumerate(chars)}
index_to_char={i:char for i,char in enumerate(chars)}
print(char_to_index)


seq_length=3
sequences=[]
labels=[] 
for i in range(len(text)-seq_length):
    seq=text[i:i+seq_length]
    label=text[i+seq_length]
    # print(seq,":::",length)
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])
    # print(list(sequences))
    # print(labels)

X=np.array(sequences)
Y=np.array(labels)
# print(X)

X_one_hot=tf.one_hot(X,len(chars))
Y_one_hot=tf.one_hot(Y,len(chars))
# print(X_one_hot)
model=Sequential()
model.add(SimpleRNN(50,input_shape=(seq_length,len(chars)),activation='relu'))
model.add(Dense(len(chars),activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_one_hot,Y_one_hot,epochs=100)