# MovieReview_SentimentAnalysis
The goal of this mini project is to implement and compare three text classification algorithms— Naive Bayes, Logistic Regression, and Multilayer Perceptron (MLP)—on the NLTK Movie Reviews dataset. To explore the impact of using both raw Term Frequency (TF) and Term Frequency- Inverse Document Frequency (TF-IDF) as feature representations.

# Sentiment Analysis of Movie Reviews using NLTK and Scikit-learn

This project explores sentiment analysis of movie reviews using the NLTK library for text preprocessing and Scikit-learn for building classification models. It covers the following aspects:

**1. Data Preparation**

* The dataset used is the NLTK movie reviews corpus.
* The code preprocesses the reviews using the following steps:
    * Tokenization: Splits reviews into individual words using the Punkt tokenizer.
    * Stemming: Reduces words to their root form using the Porter stemmer.
    * Stop Word Removal: Removes common words (like "the", "a", "is") that do not contribute much to the sentiment.
* The preprocessed reviews are converted into numerical features using:
    * Term Frequency (TF): Represents the frequency of each word in a review.
    * Term Frequency-Inverse Document Frequency (TF-IDF): Accounts for the importance of words across the entire corpus.

**2. Coverage Analysis**

* Analysis of the vocabulary coverage to determine the optimal number of tokens to use as features.
* This helps in reducing dimensionality while maintaining significant information.
* The code generates a plot showing the relationship between the number of tokens and vocabulary coverage.

**3. Sentiment Classification**

* Three different classification models are trained and evaluated:
    * Naive Bayes
    * Logistic Regression
    * Multi-Layer Perceptron (MLP)
* The models are trained using both TF and TF-IDF features to compare their performance.
* Evaluation metrics include accuracy, true positive rate (TPR), and false positive rate (FPR).

**4. MLP Architecture Exploration**

* This section involves experimenting with different MLP architectures by varying the number of layers and neurons per layer.
* The goal is to find an architecture that optimizes performance.
* Accuracy and learning curves are visualized for each architecture and feature type.

**5. Visualization**

* Results are visualized using Seaborn and Matplotlib to compare model performance.
* Bar plots are used to compare accuracy across models and feature types.
* Scatter plots visualize TPR against FPR.
* Detailed results (accuracy, TPR, FPR) are printed.

**Running the code**

1. Install necessary libraries:
   !pip install nltk scikit-learn matplotlib seaborn pandas

2. Download NLTK resources:
   import nltk
   nltk.download('punkt_tab')
   nltk.download('movie_reviews')
   nltk.download('punkt')
   nltk.download('stopwords')

3. Run the notebook code in sequence

## License

MIT License (https://opensource.org/licenses/MIT). 
