# Language Translator
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense

with open('fra1.txt','r',encoding='utf-8') as f:
    lines=f.read().split('\n')

input_texts=[]
target_texts=[]
input_characters=set()
target_characters=set()

for line in lines:
    if '\t' not in line:
        continue
    input_text,target_text=line.split('\t')
    target_text='\t'+target_text+"\n"
    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if chr not in target_characters:
            target_characters.add(char)

input_characters=sorted(list(input_characters))
target_characters=sorted(list(target_characters))

num_encoder_tokens=len(input_characters)
num_decoder_tokens=len(target_characters)

max_encoder_seq_length=max([len(txt) for txt in input_texts])
max_decoder_seq_length=max([len(txt) for txt in target_texts])

input_token_index={char: i for i,char in enumerate(input_characters)}
target_token_index={char: i for i,char in enumerate(target_characters)}

encoder_input_data=np.zeros((len(input_texts),max_encoder_seq_length,num_encoder_tokens),dtype='float32')
decoder_input_data=np.zeros((len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32')
decoder_target_data=np.zeros((len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32')


for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
    for t,char in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[char]]=1.0

    for t,char in enumerate(target_text):
        decoder_input_data[i,t,target_token_index[char]]=1.0
        if t>0:
            decoder_target_data[i,t-1,target_token_index[char]]=1.0

# Define Encoder
latent_dim=256
encoder_inputs=Input(shape=(None,num_encoder_tokens))
encoder=LSTM(latent_dim,return_state=True)
encoder_outputs,state_h,state_c=encoder(encoder_inputs)
encoder_states=[state_h,state_c]

# Define Decoder
decoder_inputs=Input(shape=(None,num_decoder_tokens))
decoder_lstm=LSTM(latent_dim,return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense=Dense(num_decoder_tokens,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)

model=Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# print(encoder_input_data)
# print(decoder_input_data)
# print(decoder_target_data)
model.fit([encoder_input_data,decoder_input_data],decoder_target_data,batch_size=64,epochs=50,validation_split=0.2)
model.save("seq2seq_translation_model.h5")