import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
import kagglehub

# Download latest version
# path = kagglehub.dataset_download("darinhawley/new-york-times-crossword-clues-answers-19932021")
# print("Path to dataset files:", path)
# path = '/Users/sheryldeakin/Desktop/CS5100/nytcrosswords.csv'

# Download dataset
path = kagglehub.dataset_download("darinhawley/new-york-times-crossword-clues-answers-19932021")
# Print all files in the dataset directory to check the correct CSV name
print("Dataset files:", os.listdir(path))
# Find the actual CSV file inside the directory
csv_filename = "nytcrosswords.csv"  # Update if needed
csv_path = os.path.join(path, csv_filename)

# Try different encodings
try:
    df = pd.read_csv(csv_path, encoding="utf-8", usecols=["Clue"], nrows=100000)  # Default UTF-8
except UnicodeDecodeError:
    print("UTF-8 failed, trying ISO-8859-1 encoding...")
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=["Clue"], nrows=100000)  # Alternative encoding

# Display the first few rows
print(df.head())

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# # Load the dataset
# # df = pd.read_csv('nyt_crossword_clues.csv')  # Adjust the filename as needed
# df = pd.read_csv(csv_path)  # Adjust the filename as needed

# # Display the first few rows of the dataset
# print('DF HEAD:', df.head())

print("Column names in the dataset:", df.columns)

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[\d{}]'.format(re.escape(string.punctuation)), '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to the 'clue' column
df['processed_clue'] = df['Clue'].apply(preprocess_text)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_clue'])

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# Determine the optimal number of clusters (K) using the Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# From the Elbow Curve, choose the optimal K (e.g., K=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_reduced)

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_k):
    cluster_points = X_reduced[df['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering of NYT Crossword Clues')
plt.legend()
plt.show()

# Analyze the clusters
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Sample Clues:")
    print(df[df['cluster'] == cluster][['clue', 'answer']].head(10))