# Sentiment Analysis on Amazon Fine Food Reviews

This project explores various techniques for sentiment analysis on the Amazon Fine Food Reviews dataset. The dataset consists of over 500,000 reviews spanning more than 10 years, including product and user information, ratings, and plain text reviews.

## Project Overview
The project aims to develop and compare different models for sentiment analysis on the Amazon Fine Food Reviews dataset. The following techniques are employed:

## Traditional Machine Learning Models:
Naive Bayes
Random Forest
Decision Tree


## Word Embeddings:
Word2Vec


## Deep Learning Models:
BERT (Bidirectional Encoder Representations from Transformers)
LoraBERT (a variant of BERT with locally randomized attention)



## Dataset
The Amazon Fine Food Reviews dataset consists of the following features:

- Product Information: Includes details about the product being reviewed.
- User Information: Contains information about the user who wrote the review.
- Rating: A numerical rating provided by the user for the product.
- Review Text: The plain text review written by the user.

## Data Preparation
The dataset underwent the following preprocessing steps:

- Loading and Handling Missing Values: The dataset was loaded, and missing values were appropriately handled.
- Binary Classification: The numerical ratings were converted into binary classes (positive and negative).
- Text Cleaning: The review text data was cleaned, including tokenization, stopword removal, lemmatization, and stemming.
- Vectorization: The cleaned text was vectorized using TF-IDF for traditional machine learning models and Word2Vec embeddings for deep learning models.

## Model Description
### Traditional Machine Learning Models
Naive Bayes, Random Forest, and Decision Tree models were trained using the TF-IDF vectorized text data.

### Word Embeddings
Word2Vec embeddings were trained on the preprocessed text data, and the embeddings were used as input features for the traditional machine learning models.

### Deep Learning Models
- BERT
  The BERT (Bidirectional Encoder Representations from Transformers) model was fine-tuned for sentiment analysis on the Amazon Fine Food Reviews dataset. The dataset was split into train and test sets, and the fine-tuning was performed using the transformers library.
- LoraBERT
LoraBERT, a variant of BERT with locally randomized attention, was employed for sentiment analysis. The model was fine-tuned similarly to BERT, and its performance was evaluated.

## Model Accuracies
The following table summarizes the accuracy, precision, recall, and F1-score for each model:

| Method        |  Precision    |        Recall |   Accuracy    | F1 Score      |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| TFIDF Naive Bayes  | 85       |            85 |            85 |            85 | 
| TFIDF Random Forest | 88      |            89 |            88 |            88 | 
| TFIDF Decision Tree | 81      |            79 |            79 |            80 |
| Word2Vec Naive Bayes| 87      |            52 |            51 |            62 | 
| Word2Vec Random Forest| 75    |            75 |            74 |            75 |
|Word2Vec Decision Tree | 69    |            69 |            68 |            69 |
| BERT (Not fine tuned) | 87    |            76 |            82 |            81 |
| BERT Fine Tuned     | 94      |            88 |            91 |            91 |
| BERT LORA           | 50      |            84 |            50 |            62 | 



## Conclusion
This project explored various techniques for sentiment analysis on the Amazon Fine Food Reviews dataset. Traditional machine learning models coupled with TF-IDF vectorization performed reasonably well, while Word2Vec embeddings yielded slightly lower accuracy. BERT, being an advanced deep learning model, outperformed traditional methods, with the fine-tuned BERT achieving the highest accuracy among all models evaluated. The findings suggest that leveraging deep learning techniques such as BERT can significantly enhance sentiment analysis tasks.
Dependencies

## The following Python libraries are required to run the code in this repository:
```
NumPy
Pandas
Scikit-learn
NLTK
Gensim
Transformers
```

You can install these dependencies using pip:
```
pip install numpy pandas scikit-learn nltk gensim transformers
```

## Usage

- Clone the repository to your local machine.
- Navigate to the cloned repository.
- Run the respective scripts or Jupyter notebooks for data preprocessing, model training, and evaluation.
- Explore the results and compare the performance of different models.
- Modify the code or experiment with different techniques as needed.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments

The Amazon Fine Food Reviews dataset used in this project is publicly available.
The project utilizes various Python libraries, including NumPy, Pandas, Scikit-learn, NLTK, Gensim, and Transformers.
