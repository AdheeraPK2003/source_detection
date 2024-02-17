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
from matplotlib import pyplot as plt
import keras
import pickle

file_path = "D:\main project\data\spam_or_not_spam.csv"
df = pd.read_csv(file_path)

spam_percentage = (df["label"].value_counts()*100/df.shape[0])[1]
ham_percentage = (df["label"].value_counts()*100/df.shape[0])[0]
print(f"Percentage of spam emails: {spam_percentage:.2f}%")

def remove_pattern(text, pattern):
    cleaned_text = re.sub(pattern, "", str(text))
    return " ".join(cleaned_text.split(" "))

df["email"] = df["email"].progress_apply(lambda x: remove_pattern(x, "NUMBER"))
df["email"] = df["email"].progress_apply(lambda text: remove_pattern(text, r'https?://\S+|www\.\S+'))
df["email"] = df["email"].progress_apply(lambda text: remove_pattern(text, "_"))
df["email"] = df["email"].progress_apply(lambda text: remove_pattern(text, r'\S+@\S+'))
df["email"] = df["email"].progress_apply(lambda text: remove_pattern(text,r'\d'))

def replace_text(text,source_pattern, destination_pattern):
    text = text.replace(source_pattern, destination_pattern)
    return text
df["email"] = df["email"].progress_apply(lambda x: replace_text(x, "won't", "will not"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x,"can't", "can not"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x, "n't", "not"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x,"'re", "are"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x,"'s", "is"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x,"'d", "would"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x,"'ll", "will"))
df["email"] = df["email"].progress_apply(lambda x: replace_text(x,"'ve", "have"))

def convert_to_lowercase(text):
    return text.lower()
df["email"] = df["email"].progress_apply(lambda x: convert_to_lowercase(x))

def replace_non_alphabets(text, replacement=' '):
    return re.sub(r'[^a-zA-Z]+', replacement, text)
df["email"] = df["email"].progress_apply(lambda x: replace_non_alphabets(x, ' '))

X= df['email']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(
  X,y , random_state=104,test_size=0.25, shuffle=True)

length_of_the_messages = df["email"].str.split("\\s+")
max_words=(length_of_the_messages.str.len().max())

keras_tokenizer=tf.keras.preprocessing.text.Tokenizer(
    num_words=max_words)
keras_tokenizer.fit_on_texts(df["email"])

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(keras_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train_sequence=keras_tokenizer.texts_to_sequences(X_train)        
X_test_sequence=keras_tokenizer.texts_to_sequences(X_test)

train_pad_sequence=pad_sequences(
X_train_sequence,
    maxlen=1000,
    padding='post',  
)
test_pad_sequence=pad_sequences(
X_test_sequence,
    maxlen=1000,
    padding='post',  
)
train_pad_sequence.shape

input_layer= Input(shape=(1000,))
dense_layer1= Dense(16)(input_layer)
dense_layer2= Dense(32)(dense_layer1)
dense_layer3= Dense(64)(dense_layer2)
dense_layer4= Dense(128)(dense_layer3)
output_layer= Dense(2)(dense_layer4)

optimizer_adam=Adam(
    learning_rate=0.001,
)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer_adam,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"]
              )
model_checkpoint_callback=keras.callbacks.ModelCheckpoint(
    filepath='D:\source_detection\model\model.hdf5',
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)
history=model.fit(
    x=train_pad_sequence,
    y=y_train,
    batch_size=32,
    epochs=25,
    validation_data=(test_pad_sequence, y_test),
    callbacks=[model_checkpoint_callback]
    )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

keras.utils.plot_model(
    model,
    to_file="D:\source_detection\plots\model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    dpi=200,
)

