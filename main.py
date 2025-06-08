import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Dataset
true = pd.read_csv("datasets/True.csv")
fake = pd.read_csv("datasets/Fake.csv")

true['label'] = 1
fake['label'] = 0
data = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)

# Clean Text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(clean_text)

# Feature Extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['text'])
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Report:\n", classification_report(y_test, pred))
