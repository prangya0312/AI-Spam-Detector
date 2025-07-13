import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltkpython spam_detector.py



# Download nltk data (only once)
nltk.download('punkt')

# Load and clean dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Evaluate
y_pred = model.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict function
def predict_sms(text):
    text_counts = vectorizer.transform([text])
    result = model.predict(text_counts)
    return "Spam" if result[0] else "Not Spam"

# Try it on a few messages
print("\nPrediction examples:")
print("1:", predict_sms("Win a FREE vacation to Maldives now!"))
print("2:", predict_sms("Hey Prangya, want to grab lunch at 2?"))
print("3:", predict_sms("Your account is blocked. Click here to reactivate."))
