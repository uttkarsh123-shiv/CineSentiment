#🎬 CineSentiment: Deep Learning-Based Sentiment Analysis

#🌟 Project Overview
This project implements a deep learning model to classify the sentiment (e.g., positive or negative) of text data, primarily focusing on cinema or movie-related content, as suggested by the project naming.

The core of the system is a Long Short-Term Memory (LSTM) neural network built using TensorFlow/Keras, which is highly effective for sequential data like natural language. The model is trained to understand the context and nuances in text to accurately determine the underlying emotional tone.

#⚙️ Model Architecture
The deep learning pipeline utilizes the following key components, visible in the included dependencies:

Tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): Converts raw text into numerical sequences, ensuring the model can process the input. The vocabulary learned during training is saved in tokenizer.pkl.

Embedding Layer: Maps each word in the vocabulary to a dense vector, allowing the model to understand semantic relationships between words.

LSTM Layer: The recurrent layer responsible for learning long-term dependencies in the text sequence, which is crucial for capturing the overall sentiment of a sentence or paragraph.

Dense Layers: Standard feed-forward layers used for final classification (e.g., outputting a probability score for positive sentiment).

#📁 Repository Structure
File

Description

CineSentiment.ipynb

Primary Training Notebook. Contains all steps for data loading, preprocessing, model definition, training, evaluation, and saving the final model (model.h5) and tokenizer (tokenizer.pkl).

Demo.ipynb

Inference and Testing Notebook. Demonstrates how to load the saved model.h5 and tokenizer.pkl files and use them to predict the sentiment of new, unseen text inputs.

model.h5

Trained Deep Learning Model. This is the final, saved weights and architecture of the Keras Sequential Model (LSTM). Ready for immediate deployment or inference.

tokenizer.pkl

Saved Keras Tokenizer. Essential for processing new text. Any new text must be processed by this exact tokenizer before being passed to model.h5 for a prediction.

README.md

This file.

🚀 How to Use the Model for Inference
To run a prediction on new data, you need both the model and the tokenizer. The following steps outline the typical process (fully demonstrated in Demo.ipynb):

Install Dependencies: Ensure you have the necessary libraries installed.

pip install tensorflow pandas scikit-learn numpy

Load Assets: Load the model and tokenizer objects.

import pickle
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

Preprocess Input: Tokenize and pad your new text input to match the sequence length used during training.

Predict: Use model.predict(padded_sequence) to get the sentiment output.

📝 Dependencies
This project requires the following major libraries:

Python (3.x)

TensorFlow / Keras

Pandas

NumPy

Scikit-learn (sklearn)
