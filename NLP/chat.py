import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK data
nltk.download('punkt')

# --- Load intents ---
with open('data.json', 'r') as file:
    data = json.load(file)

# --- Prepare training data ---
sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# --- NLP preprocessing ---
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(sentences)
y = labels

# --- Train a simple classifier ---
model = MultinomialNB()
model.fit(X, y)

# --- Chat function ---
def chatbot_response(text):
    X_test = vectorizer.transform([text])
    predicted_tag = model.predict(X_test)[0]

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

    return "I'm not sure I understand. Could you rephrase?"

# --- Run chatbot ---
print("🤖 Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
