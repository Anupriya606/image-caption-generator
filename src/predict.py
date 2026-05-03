import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Load model
print("Loading caption model...")
model = load_model(MODEL_PATH)
print("Caption model loaded!")

# Load feature extractor
print("Loading InceptionV3...")
base_model = InceptionV3(weights='imagenet')
feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)
print("InceptionV3 loaded!")

# Key parameters
max_length = 37
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

def extract_features(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = feature_extractor.predict(img, verbose=0)
    return feature

def generate_caption(image_path):
    feature = extract_features(image_path)
    caption = 'startseq'
    
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        pred = model.predict([feature, seq], verbose=0)
        pred_idx = np.argmax(pred)
        word = index_to_word.get(pred_idx, None)
        if word is None or word == 'endseq':
            break
        caption += ' ' + word
    
    caption = caption.replace('startseq', '').strip()
    return caption