# Install for 
# pip install keras
# pip install tensorflow
# pip install tf-keras
# pip install transformers
# pip install nltk
# pip install sentencepiece
# pip install gradio

import gradio as gr
from transformers import pipeline

#English to Dutch 

# translation_pipeline=pipeline("translation_en_to_de")
# data="My Name is Naman Jain"
# results=translation_pipeline(data)
# print("In English::",data)
# print("In Dutch::",results)

#English to Hindi

translation_pipeline=pipeline("translation_en_to_hi",model="Helsinki-NLP/opus-mt-en-hi")
data="My Name is Naman Jain"

def translate_transformers(from_data):
    results=translation_pipeline(from_data)
    res=results[0]['translation_text']
    return res
translate_transformers(data)

# create gradio
translate=gr.Interface(inputs=gr.Textbox(lines=2,placeholder="Text to Translate"),fn=translate_transformers,outputs='text')
translate.launch(share=False)
# print("In English::",data)
# print("In Hindi::",translate)