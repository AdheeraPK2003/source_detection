import streamlit as st
import pandas as pd
import re
import pickle
from keras.layers import Input
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

def remove_pattern(text, pattern):
    cleaned_text = re.sub(pattern, "", str(text))
    return " ".join(cleaned_text.split(" "))

def replace_text(text,source_pattern, destination_pattern):
     text = text.replace(source_pattern, destination_pattern)
     return text

with open('tokenizer.pkl', 'rb') as handle:
    keras_tokenizer = pickle.load(handle)


st.write("""
# Spam Email Classifier
""")
 


email_content = st.text_input('')

email_content = remove_pattern(email_content, "NUMBER")
email_content = remove_pattern(email_content, r"https?://\S+|www\.\S+'")
email_content = remove_pattern(email_content, r"\S+@\S+")
email_content = remove_pattern(email_content, r"'\d'")
email_content = remove_pattern(email_content, "_")

email_content = replace_text(email_content, "won't","will not")
email_content = replace_text(email_content, "can't", "can not")
email_content = replace_text(email_content, "n't", "not" )
email_content = replace_text(email_content, "'re", "are" )
email_content = replace_text(email_content, "'s", "is")
email_content = replace_text(email_content, "'d", "would")
email_content = replace_text(email_content, "'ll", "will")
email_content = replace_text(email_content, "'ve", "have")
st.write('entered text is \n\r', email_content)

test_sequense=keras_tokenizer.texts_to_sequences([email_content])
test_sequence=pad_sequences(test_sequense,
    maxlen=1000,
    padding='post',  
)

input_layer= Input(shape=(1000,))
dense_layer1= Dense(16, activation='relu')(input_layer)
dense_layer2= Dense(32, activation='relu')(dense_layer1)
dense_layer3= Dense(64, activation='relu')(dense_layer2)
dense_layer4= Dense(128, activation='relu')(dense_layer3)
output_layer= Dense(2, activation='softmax')(dense_layer4)

optimizer_adam=Adam(
    learning_rate=0.001,
)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer_adam,
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"]
              )

model.build(input_shape=(1000,))
model.load_weights('D:\source_detection\model\model.hdf5')
y_pred = model.predict([test_sequence], verbose=0)

st.button("Predict", type="primary")

print(y_pred)