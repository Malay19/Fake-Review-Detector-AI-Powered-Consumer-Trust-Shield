
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords


# Load dataset
df = pd.read_csv('Preprocessed Fake Reviews Detection Dataset.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.dropna(inplace=True)

# Add text length feature
df['length'] = df['text_'].apply(len)

# Exploratory Data Analysis
plt.hist(df['length'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()

df.groupby('label').describe()

# Ensure stopwords are available
nltk.download('stopwords')

def text_process(review):
    stop_words = set(stopwords.words('english'))  # Load stopwords only once
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stop_words]


warnings.filterwarnings('ignore')
# Text preprocessing function
def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Convert text to numerical representation
bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer.fit(df['text_'])

bow_reviews = bow_transformer.transform(df['text_'])
tfidf_transformer = TfidfTransformer().fit(bow_reviews)
tfidf_reviews = tfidf_transformer.transform(bow_reviews)

# Split dataset
review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=0.35)

# List of models to evaluate
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN (k=2)": KNeighborsClassifier(n_neighbors=2),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', model)
    ])

    pipeline.fit(review_train, label_train)
    predictions = pipeline.predict(review_test)

    acc = accuracy_score(label_test, predictions)
    results[model_name] = acc

    print(f"\nModel: {model_name}")
    print('------------------------------------')
    print('Classification Report:\n', classification_report(label_test, predictions))
    print('Confusion Matrix:\n', confusion_matrix(label_test, predictions))
    print(f'Accuracy: {acc * 100:.2f}%\n')

# Feature Importance Analysis (For tree-based models)
if "Random Forest" in models:
    rf_model = models["Random Forest"]
    rf_model.fit(bow_reviews, df['label'])
    feature_importances = rf_model.feature_importances_

    # Get top 10 most important words
top_indices = np.argsort(feature_importances)[-10:]
top_words = [bow_transformer.get_feature_names_out()[i] for i in top_indices]

print("\nTop 10 words influencing classification:")
for word in top_words:
    print(word)


# Outcome Summary
print("\nFinal Model Performance Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc * 100:.2f}% accuracy")

# Fake Review Analysis: Common Words
fake_reviews = df[df['label'] == 'FR']['text_']
genuine_reviews = df[df['label'] == 'OR']['text_']

fake_word_freq = pd.Series(' '.join(fake_reviews).split()).value_counts().head(10)
genuine_word_freq = pd.Series(' '.join(genuine_reviews).split()).value_counts().head(10)

print("\nMost common words in Fake Reviews:", fake_word_freq)
print("\nMost common words in Genuine Reviews:", genuine_word_freq)
