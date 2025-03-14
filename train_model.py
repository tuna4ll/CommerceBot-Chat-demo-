import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
import pickle
import random
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

vocab_size = 10000  
embedding_dim = 64 
max_len = 30  
oov_tok = "<OOV>"  

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))  
model.add(Dense(32, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

model.fit(padded_sequences, np.array(training_labels_encoded), epochs=300, batch_size=8)

model.save("chatbot_model.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(lbl_encoder, open("label_encoder.pkl", "wb"))
