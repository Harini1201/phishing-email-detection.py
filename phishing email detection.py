import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


data = {
    "email": [
        "Congratulations! You won a free iPhone. Click here http://fake.com",
        "Your bank account is suspended. Verify immediately.",
        "Meeting scheduled tomorrow at 10 AM.",
        "Project report attached. Please review.",
        "Claim your lottery prize now by visiting http://scam.com",
        "Team lunch planned for Friday.",
        "Update your password immediately to avoid account closure.",
        "Thank you for your purchase. Receipt attached."
    ],
    "label": [
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Safe",
        "Phishing",
        "Safe"
    ]
}

df = pd.DataFrame(data)


def extract_features(text):
    urls = len(re.findall(r'http[s]?://\S+', text))

    suspicious_keywords = [
        "verify", "click", "password",
        "account", "bank", "lottery",
        "winner", "free", "claim"
    ]

    keyword_count = sum(
        keyword in text.lower()
        for keyword in suspicious_keywords
    )

    email_length = len(text)

    return pd.Series([urls, keyword_count, email_length])

df[['url_count', 'keyword_count', 'email_length']] = df['email'].apply(extract_features)


vectorizer = TfidfVectorizer(stop_words='english')
text_features = vectorizer.fit_transform(df['email'])


extra_features = df[['url_count', 'keyword_count', 'email_length']].values

from scipy.sparse import hstack
X = hstack([text_features, extra_features])

y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ["Phishing", "Safe"])
plt.yticks([0, 1], ["Phishing", "Safe"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.show()


while True:
    email = input("\nEnter Email (or type exit): ")

    if email.lower() == "exit":
        break

    text_feature = vectorizer.transform([email])

    url_count = len(re.findall(r'http[s]?://\S+', email))

    suspicious_keywords = [
        "verify", "click", "password",
        "account", "bank", "lottery",
        "winner", "free", "claim"
    ]

    keyword_count = sum(
        keyword in email.lower()
        for keyword in suspicious_keywords
    )

    email_length = len(email)

    extra_feature = [[url_count, keyword_count, email_length]]

    test_data = hstack([text_feature, extra_feature])

    prediction = model.predict(test_data)

    print("Result:", prediction[0])