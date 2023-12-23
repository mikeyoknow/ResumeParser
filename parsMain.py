import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('path_to_dataset.csv')
print(df.head())

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')
              and word.isalpha()]
    return ' '.join(tokens)

df['cleaned_resume'] = df['resume_column_name'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=1000)  
tfidf_matrix = vectorizer.fit_transform(df['cleaned_resume'])
