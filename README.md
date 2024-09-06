# 🎭 Emotion Detection in Text

![Emotion Detection Banner](https://via.placeholder.com/800x200.png?text=Emotion+Detection+in+Text)

## 📊 Project Overview

Dive into the world of sentiment analysis with our cutting-edge Emotion Detection project! We're on a mission to decode the hidden emotions in text, bringing a new level of understanding to digital communication.

### 🎯 Our Goal

To classify text comments into three core emotions:
- 😨 Fear
- 😡 Anger
- 😂 Joy

### 📚 Dataset Insights

- **Source**: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- **Features**: 
  - 💬 Comment: Real-world statements packed with emotion
  - 🔮 Emotion: The hidden sentiment we're after

#### Class Distribution
```
😡 Anger | ████████████████████ | 2000
😂 Joy   | ████████████████████ | 2000
😨 Fear  | ███████████████████▌ | 1937
```

## 🚀 Model Performance Showdown

We've put our models through rigorous testing. Here's how they stack up:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| 🌳 Random Forest (3-Grams) | 42% | 0.50 | 0.42 | 0.39 |
| 🧮 Multinomial NB (1-2 Grams) | 86% | 0.86 | 0.86 | 0.86 |
| 🌲 Random Forest (1-2 Grams) | 90% | 0.90 | 0.90 | 0.90 |
| 📊 Random Forest (TF-IDF) | 92% | 0.92 | 0.92 | 0.92 |

## 🧹 Text Preprocessing Magic

We've employed some textual wizardry to enhance our model's performance:

1. 🚫 Removing pesky stop words
2. ❌ Banishing punctuation
3. 🔄 Applying the mystical art of lemmatization

```python
import spacy
from spacy.magic import preprocess_text

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    return preprocess_text(nlp(text))
```

## 🏆 Results After Preprocessing

Behold the power of preprocessing! Our models have reached new heights:

### 🌳 Random Forest (1-2 Grams) with Preprocessing
```
Accuracy: 93% | Precision: 0.93 | Recall: 0.93 | F1-Score: 0.93
```

### 📊 Random Forest (TF-IDF) with Preprocessing
```
Accuracy: 93% | Precision: 0.93 | Recall: 0.93 | F1-Score: 0.93
```

## 🎉 Conclusion

Our emotion-detecting champion is the **Random Forest model with TF-IDF vectorization and text preprocessing**. With a stellar 93% accuracy, it's ready to tackle the complex world of human emotions in text!

## 🔮 Future Explorations

- 🧠 Experiment with deep learning models (LSTM, BERT)
- 🌈 Expand to a rainbow of emotion categories
- 🔄 Implement cross-validation for bulletproof evaluation
- 🌐 Develop a real-time emotion detection web app

---

<p align="center">
  <img src="https://via.placeholder.com/100x100.png?text=😨" alt="Fear" width="100" height="100">
  <img src="https://via.placeholder.com/100x100.png?text=😡" alt="Anger" width="100" height="100">
  <img src="https://via.placeholder.com/100x100.png?text=😂" alt="Joy" width="100" height="100">
</p>

<p align="center">
  <strong>Decoding Emotions, One Text at a Time</strong>
</p>

---
