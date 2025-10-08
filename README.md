# 🎬 CineSentiment: Deep Learning-Based Sentiment Analysis

---

## 🌟 Project Overview
*Classify the sentiment of text data, especially movie-related content, using deep learning.*  

This project uses a **Long Short-Term Memory (LSTM)** model built with **TensorFlow/Keras**, designed to understand the context and nuances of text for accurate sentiment prediction.

---

## ⚙️ Model Architecture
- **Tokenizer** (`tensorflow.keras.preprocessing.text.Tokenizer`): Converts raw text to numerical sequences. Saved as `tokenizer.pkl`.  
- **Embedding Layer**: Maps words to dense vectors to capture semantic meaning.  
- **LSTM Layer**: Learns long-term dependencies in text sequences for context-aware sentiment detection.  
- **Dense Layers**: Feed-forward layers for final classification (positive/negative sentiment).

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `CineSentiment.ipynb` | Training notebook: data loading, preprocessing, model creation, training, evaluation, and saving `model.h5` & `tokenizer.pkl`. |
| `Demo.ipynb` | Inference notebook: demonstrates how to load and predict sentiment using saved assets. |
| `model.h5` | Trained LSTM model ready for deployment. |
| `tokenizer.pkl` | Saved tokenizer for consistent preprocessing of new text inputs. |
| `README.md` | Project overview and usage instructions. |

---

## 🚀 How to Use the Model for Inference

### 1️⃣ Install Dependencies
```bash
pip install tensorflow pandas scikit-learn numpy
