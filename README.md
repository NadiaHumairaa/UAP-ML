# UAP-ML
# README

## Deskripsi Proyek
Sistem ini dirancang untuk menganalisis sentimen ulasan restoran dengan menggunakan metode pembelajaran mesin dan pembelajaran mendalam. Proyek ini bertujuan untuk mempermudah pemilik restoran memahami umpan balik pelanggan, meningkatkan kualitas layanan, dan mendeteksi ulasan positif atau negatif secara otomatis.

---

## Langkah Instalasi

### 1. Instalasi Dependencies
Pastikan Anda memiliki Python 3.x terinstal. Lakukan instalasi pustaka yang diperlukan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Web
1. Jalankan skrip Flask dengan perintah:
```bash
python app.py
```
2. Buka browser Anda dan akses aplikasi melalui `http://127.0.0.1:5000/`.

---

## Deskripsi Model

### Model yang Digunakan
1. **Random Forest Classifier**:
   - Digunakan untuk memprediksi rating ulasan.
   - Menggunakan fitur representasi teks TF-IDF.

2. **Deep Learning Models**:
   - **LSTM** (Long Short-Term Memory)
   - **GRU** (Gated Recurrent Unit)
   - **Transformer**
   - Ketiga model ini digunakan untuk klasifikasi sentimen ulasan dengan representasi teks yang diproses menggunakan Tokenizer dan Padding.

### Analisis Performa
Model dievaluasi menggunakan metrik akurasi dan F1-score. Performa model ditampilkan dalam tabel berikut:

| Model         | Accuracy | F1-Score |
|---------------|----------|----------|
| LSTM          | X.XX     | X.XX     |
| GRU           | X.XX     | X.XX     |
| Transformer   | X.XX     | X.XX     |
| Random Forest | X.XX     | N/A      |

---

## Hasil dan Analisis
Model deep learning seperti LSTM, GRU, dan Transformer menunjukkan kemampuan yang baik dalam mendeteksi sentimen ulasan. Tabel di atas menunjukkan perbandingan akurasi dan F1-score antara model pembelajaran mesin dan pembelajaran mendalam.

Tambahkan visualisasi grafik hasil evaluasi dengan menggunakan pustaka seperti Matplotlib atau Seaborn untuk memperjelas perbandingan performa.

---

## Link Live Demo
[Aplikasi Live Demo](#) (tambahkan tautan ke aplikasi yang telah dideploy).

---

## Struktur Proyek
```
.
├── McDonald_s_Reviews.csv       # Dataset asli
├── processed_reviews.csv        # Dataset setelah diproses
├── preprocess.py                # Skrip untuk preprocessing data
├── train_ml_model.py            # Skrip pelatihan model pembelajaran mesin
├── train_dl_models.py           # Skrip pelatihan model pembelajaran mendalam
├── app.py                       # Aplikasi web Flask
├── restaurant_review_model.pkl  # Model Random Forest yang disimpan
├── tfidf_vectorizer.pkl         # Vectorizer TF-IDF yang disimpan
├── templates/
│   ├── index.html               # Template form input
│   ├── result.html              # Template hasil prediksi
└── README.md                    # Dokumentasi proyek
```

---

## Acknowledgments
- **NLTK** untuk preprocessing bahasa alami.
- **TensorFlow** untuk implementasi model pembelajaran mendalam.
- **Scikit-learn** untuk model pembelajaran mesin.
- **Flask** untuk pengembangan aplikasi web.

---

## Catatan Tambahan
Untuk detail lebih lanjut mengenai implementasi setiap model, buka file skrip terkait di direktori proyek ini.
