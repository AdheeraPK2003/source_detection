import re
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Input
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Model

file_path = "D:\main project\data\spam_or_not_spam.csv"
df = pd.read_csv(file_path)

spam_percentage = (df["label"].value_counts()*100/df.shape[0])[1]
ham_percentage = (df["label"].value_counts()*100/df.shape[0])[0]
print(f"Percentage of spam emails: {spam_percentage:.2f}%")

def remove_pattern(text, pattern):
    cleaned_text = re.sub(pattern, "", str(text))
    return " ".join(cleaned_text.split(" "))

df["email"] = df["email"].progress_apply(lambda x: remove_pattern(x, "NUMBER"))
df["email"] = df["email"].progress_apply(lambda text: re.sub(r'https?://\S+|www\.\S+', '', text))
df["email"] = df["email"].progress_apply(lambda text: text.replace("_", ""))
df["email"] = df["email"].progress_apply(lambda text: re.sub(r'\S+@\S+', '', text))
df["email"] = df["email"].progress_apply(lambda text: re.sub(r'\d', '', text))

X= df['email']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(
  X,y , random_state=104,test_size=0.25, shuffle=True)

length_of_the_messages = df["email"].str.split("\\s+")
max_words=(length_of_the_messages.str.len().max())

keras_tokenizer=tf.keras.preprocessing.text.Tokenizer(
    num_words=max_words)
keras_tokenizer.fit_on_texts(df["email"])
X_train_sequence=keras_tokenizer.texts_to_sequences(X_train)        
X_test_sequence=keras_tokenizer.texts_to_sequences(X_test)

train_pad_sequence=pad_sequences(
X_train_sequence,
    maxlen=10000,
    padding='post',  
)
test_pad_sequence=pad_sequences(
X_test_sequence,
    maxlen=10000,
    padding='post',  
)
train_pad_sequence.shape

input_layer= Input(shape=(10000,))
dense_layer1= Dense(16)(input_layer)
output_layer= Dense(2)(dense_layer1)

optimizer_adam=Adam(
    learning_rate=0.001,
)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer_adam,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"]
              )
model.fit(
    x=train_pad_sequence,
    y=y_train,
    batch_size=32,
    epochs=50,
    )