import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

df = pd.read_csv("McDonald_s_Reviews.csv", encoding='ISO-8859-1')

if 'review' not in df.columns:
    print("Kolom 'review' tidak ditemukan dalam dataset.")
else:
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        # Menghapus karakter tidak valid seperti ï¿½ï¿½ï¿½ï¿½ï¿½ï¿
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Mengubah teks menjadi huruf kecil
        text = text.lower()

        # Menghapus karakter selain huruf dan spasi
        text = re.sub(r'[^a-z\s]', '', text)

        # Menghapus stopwords
        text = ' '.join([word for word in text.split() if word not in stop_words])

        # Melakukan stemming
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])

        return text

    # Memproses kolom 'review'
    df['processed_review'] = df['review'].apply(preprocess)

    # Menyimpan hasil preprocessing ke file baru
    df.to_csv("processed_reviews.csv", index=False)

    # Menampilkan hasil preprocessing
    print(df[['review', 'processed_review']].head())
