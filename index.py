from flask import Flask, render_template, request, jsonify
import json
import random
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("chatbot_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
lbl_encoder = pickle.load(open("label_encoder.pkl", "rb"))

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

max_len = 20

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["message"]
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([msg]), truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for intent in data["intents"]:
        if intent["tag"] == tag[0]:
            return jsonify({"response": random.choice(intent["responses"])})

    return jsonify({"response": "Üzgünüm, bunu anlayamadım. Başka bir şekilde ifade edebilir misiniz?"})

if __name__ == "__main__":
    app.run(debug=True)
