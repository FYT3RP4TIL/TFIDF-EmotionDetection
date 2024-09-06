<h1 align="center">
  🎭 Emotion Detection in Text:<br>
  A Deep Dive into Sentiment Analysis<br>
  Via TFIDF and N-Grams
</h1>

## 📊 Project Overview

Welcome to our advanced Emotion Detection project! This comprehensive study delves into the intricate world of sentiment analysis, employing cutting-edge Natural Language Processing (NLP) techniques and machine learning models to decode the subtle nuances of human emotions expressed in text.

### 🎯 Project Objectives

1. Develop robust models for accurately classifying text into three primary emotions:
   - 😨 Fear: Anticipation of threat or danger
   - 😡 Anger: Strong feeling of annoyance, displeasure, or hostility
   - 😂 Joy: Feeling of great pleasure and happiness
2. Compare and contrast various NLP techniques and machine learning algorithms
3. Explore the impact of text preprocessing on model performance
4. Create a foundation for more advanced emotion detection systems

## 📚 Dataset: The Foundation of Our Analysis

### Dataset Source and Description

Our project utilizes the [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp), a carefully curated collection of text samples paired with corresponding emotions.

#### Key Features:

1. 💬 **Comment**: 
   - Real-world statements and messages related to various events and situations
   - Diverse in length, complexity, and subject matter
   - Represents natural language usage across different contexts

2. 🔮 **Emotion**: 
   - The labeled emotion associated with each comment
   - Limited to three primary categories: fear, anger, and joy
   - Provides a balanced representation of each emotion

### Dataset Statistics

#### Class Distribution
```
😡 Anger | ████████████████████ | 2000 samples | 33.73%
😂 Joy   | ████████████████████ | 2000 samples | 33.73%
😨 Fear  | ███████████████████▌ | 1937 samples | 32.54%
```

Total samples: 5,937

#### Data Quality
- Balanced distribution ensures unbiased model training
- Large sample size provides robust training and evaluation capabilities
- Real-world text data captures the complexity of natural language expression

## 🚀 Model Development and Evaluation

We implemented and evaluated several models, each with unique characteristics and performance profiles. Here's an in-depth look at our model lineup:

### 1. 🌳 Random Forest with 3-Grams

#### Configuration:
- Vectorization: CountVectorizer with ngram_range=(3, 3)
- Model: RandomForestClassifier with default parameters

#### Performance:
```
              precision    recall  f1-score   support
           0       0.58      0.26      0.36       400
           1       0.37      0.80      0.51       388
           2       0.53      0.22      0.31       400
    accuracy                           0.42      1188
   macro avg       0.49      0.43      0.39      1188
weighted avg       0.50      0.42      0.39      1188
```

#### Analysis:
- Lower overall performance compared to other models
- High recall for class 1 (0.80) but poor precision (0.37)
- Struggles with classes 0 and 2, indicating potential overfitting to class 1

### 2. 🧮 Multinomial Naive Bayes with 1-2 Grams

#### Configuration:
- Vectorization: CountVectorizer with ngram_range=(1, 2)
- Model: MultinomialNB with default parameters

#### Performance:
```
              precision    recall  f1-score   support
           0       0.87      0.86      0.87       400
           1       0.87      0.83      0.85       388
           2       0.83      0.88      0.85       400
    accuracy                           0.86      1188
   macro avg       0.86      0.86      0.86      1188
weighted avg       0.86      0.86      0.86      1188
```

#### Analysis:
- Significant improvement over the 3-gram Random Forest model
- Balanced performance across all classes
- Good overall accuracy of 86%

### 3. 🌲 Random Forest with 1-2 Grams

#### Configuration:
- Vectorization: CountVectorizer with ngram_range=(1, 2)
- Model: RandomForestClassifier with default parameters

#### Performance:
```
              precision    recall  f1-score   support
           0       0.83      0.96      0.89       400
           1       0.95      0.87      0.91       388
           2       0.93      0.87      0.90       400
    accuracy                           0.90      1188
   macro avg       0.90      0.90      0.90      1188
weighted avg       0.90      0.90      0.90      1188
```

#### Analysis:
- Further improvement in overall performance
- High precision and recall across all classes
- Particularly strong in identifying class 1 (0.95 precision)

### 4. 📊 Random Forest with TF-IDF

#### Configuration:
- Vectorization: TfidfVectorizer with default parameters
- Model: RandomForestClassifier with default parameters

#### Performance:
```
              precision    recall  f1-score   support
           0       0.89      0.95      0.92       400
           1       0.92      0.91      0.92       388
           2       0.94      0.88      0.91       400
    accuracy                           0.92      1188
   macro avg       0.92      0.92      0.92      1188
weighted avg       0.92      0.92      0.92      1188
```

#### Analysis:
- Best performing model before preprocessing
- Excellent balance of precision and recall across all classes
- TF-IDF vectorization appears to capture important features effectively

## 🧹 Text Preprocessing: Enhancing Model Input

To further improve our models' performance, we implemented a comprehensive text preprocessing pipeline. This crucial step helps to normalize the input data, reduce noise, and focus on the most meaningful aspects of the text.

### Preprocessing Steps:

1. **Removing Stop Words**
   - Eliminates common words (e.g., "the", "is", "at") that typically don't carry significant emotional content
   - Helps models focus on more meaningful words

2. **Removing Punctuation**
   - Strips away punctuation marks to standardize text input
   - Reduces noise and potential inconsistencies in punctuation usage

3. **Applying Lemmatization**
   - Reduces words to their base or dictionary form
   - Helps consolidate different forms of a word (e.g., "running", "ran", "runs" → "run")
   - Maintains the core meaning of words better than simple stemming

### Implementation:

We utilized the powerful spaCy library for our preprocessing pipeline:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens)
```

This function processes each input text by:
1. Tokenizing the text using spaCy's linguistic model
2. Filtering out stop words and punctuation
3. Lemmatizing the remaining tokens
4. Joining the processed tokens back into a single string

## 🏆 Results After Preprocessing

The application of our preprocessing pipeline led to significant improvements in model performance:

### 5. 🌳 Random Forest (1-2 Grams) with Preprocessing

#### Configuration:
- Preprocessing: Custom pipeline (stop words removal, punctuation removal, lemmatization)
- Vectorization: CountVectorizer with ngram_range=(1, 2)
- Model: RandomForestClassifier with default parameters

#### Performance:
```
              precision    recall  f1-score   support
           0       0.94      0.95      0.95       400
           1       0.94      0.91      0.93       388
           2       0.93      0.93      0.93       400
    accuracy                           0.93      1188
   macro avg       0.93      0.93      0.93      1188
weighted avg       0.93      0.93      0.93      1188
```

#### Analysis:
- Substantial improvement over the non-preprocessed version
- High and balanced precision and recall across all classes
- Preprocessing appears to have helped in distinguishing between emotions more effectively

### 6. 📊 Random Forest (TF-IDF) with Preprocessing

#### Configuration:
- Preprocessing: Custom pipeline (stop words removal, punctuation removal, lemmatization)
- Vectorization: TfidfVectorizer with default parameters
- Model: RandomForestClassifier with default parameters

#### Performance:
```
              precision    recall  f1-score   support
           0       0.92      0.96      0.94       400
           1       0.92      0.92      0.92       388
           2       0.94      0.90      0.92       400
    accuracy                           0.93      1188
   macro avg       0.93      0.93      0.93      1188
weighted avg       0.93      0.93      0.93      1188
```

#### Analysis:
- Matches the performance of the 1-2 Grams model with preprocessing
- Slight improvements in certain class-specific metrics
- Demonstrates the robust performance of Random Forest with TF-IDF, even with preprocessing

## 🎉 Conclusion and Key Findings

After extensive experimentation and analysis, we can draw several important conclusions:

1. **Preprocessing Impact**: The application of our custom preprocessing pipeline consistently improved model performance, highlighting the importance of text normalization in emotion detection tasks.

2. **Best Performing Model**: The Random Forest model, whether using 1-2 Grams or TF-IDF vectorization, combined with our preprocessing pipeline, achieved the best overall performance with 93% accuracy.

3. **Feature Representation**: Both Count Vectorization (with 1-2 Grams) and TF-IDF Vectorization proved effective in capturing relevant features for emotion detection.

4. **Balanced Performance**: Our top models demonstrated balanced precision and recall across all three emotion classes, indicating robust and reliable classification capabilities.

5. **Model Complexity**: The Random Forest algorithm consistently outperformed simpler models like Multinomial Naive Bayes, suggesting that the complexity of emotion detection benefits from ensemble methods.

## 🔮 Future Directions and Potential Enhancements

While our current models have achieved impressive results, there are several exciting avenues for further research and improvement:

1. **Deep Learning Exploration**: 
   - Implement and evaluate deep learning models such as LSTM (Long Short-Term Memory) or BERT (Bidirectional Encoder Representations from Transformers)
   - Explore the potential of transfer learning using pre-trained language models

2. **Expanded Emotion Categories**: 
   - Extend the model to classify a broader range of emotions (e.g., surprise, disgust, sadness)
   - Investigate multi-label classification for texts expressing multiple emotions

3. **Advanced Cross-Validation**: 
   - Implement k-fold cross-validation for more robust model evaluation
   - Explore stratified sampling techniques to ensure balanced representation of emotions in all folds

4. **Real-Time Application Development**: 
   - Develop a web application or API for real-time emotion detection in text
   - Integrate the emotion detection system with chatbots or social media analysis tools

5. **Feature Importance Analysis**: 
   - Conduct in-depth analysis of feature importance to understand key indicators of different emotions
   - Use this insight to further refine preprocessing and feature selection techniques

6. **Error Analysis**: 
   - Perform detailed error analysis to identify common misclassifications
   - Use these insights to develop targeted improvements in preprocessing or model architecture

7. **Multilingual Expansion**: 
   - Extend the emotion detection capabilities to multiple languages
   - Investigate cross-lingual emotion detection techniques

By pursuing these directions, we aim to push the boundaries of emotion detection in text, contributing to the broader field of affective computing and natural language understanding.
