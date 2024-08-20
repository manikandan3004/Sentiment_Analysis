import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Suppress all warnings
warnings.filterwarnings("ignore")

# Step 1: Load and clean the dataset
file_path = 'sentimentdataset.csv'
data = pd.read_csv(file_path)

# Remove unnecessary columns
data_cleaned = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

# Step 2: Preprocess the text data
# Select the relevant columns
texts = data_cleaned['Text']
sentiments = data_cleaned['Sentiment'].str.strip()  # Remove any leading/trailing spaces

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 3: Train a sentiment classification model
# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict the sentiments of the test set
y_pred = model.predict(X_test_tfidf)

# Step 4: Analyze the model's performance
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display a detailed classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Step 5: Classify new text data (optional)
# Example of classifying new text data
new_texts = ["I love this product!", "This is the worst experience ever.", "I'm not sure how I feel about this."]
new_texts_tfidf = vectorizer.transform(new_texts)
predicted_sentiments = model.predict(new_texts_tfidf)

for text, sentiment in zip(new_texts, predicted_sentiments):
    print(f"Text: {text} => Sentiment: {sentiment}")
