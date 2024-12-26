# Analisis Sentimen Ulasan McDonald's untuk Menilai Kepuasan Konsumen

## Deskripsi
Proyek ini bertujuan untuk menganalisis sentimen ulasan pelanggan McDonald's dari platform Google Reviews. Analisis ini menggunakan teknik machine learning dan deep learning untuk memproses ulasan, mengkategorikannya ke dalam tiga kelas utama berdasarkan rating: **positif**, **netral**, dan **negatif**. Sistem ini juga menyediakan antarmuka berbasis web untuk memprediksi sentimen dari ulasan baru secara real-time.

## Dataset
Dataset yang digunakan adalah file **McDonald_s_Reviews.csv**, yang berisi ulasan pelanggan McDonald's di Amerika Serikat. Beberapa kolom penting dalam dataset:
- **review**: Teks ulasan pelanggan.
- **rating**: Rating yang diberikan pelanggan (1–5 bintang).

### Preprocessing
1. **Penghapusan karakter non-alfabetik**: Semua karakter selain huruf dihapus dari teks.
2. **Lowercasing**: Semua teks diubah menjadi huruf kecil.
3. **Stopword Removal**: Kata-kata umum yang tidak memberikan banyak informasi (seperti "the", "is") dihapus.
4. **Stemming**: Kata-kata dikembalikan ke bentuk dasarnya menggunakan algoritma Porter Stemmer.

Hasil preprocessing disimpan dalam file **processed_reviews.csv**.

## Deskripsi Model
Proyek ini menggunakan berbagai teknik dan model untuk analisis sentimen:
1. **Deep Learning Models**:
   - **LSTM (Long Short-Term Memory)**: Model berbasis RNN untuk memahami konteks dalam urutan teks.
   - **GRU (Gated Recurrent Unit)**: Alternatif RNN yang lebih efisien untuk urutan panjang.
   - **Transformer**: Model canggih dengan mekanisme perhatian untuk memahami relasi antar kata.
2. **Flask Web App**:
   - Antarmuka pengguna untuk memasukkan ulasan baru dan mendapatkan prediksi sentimen.

## Hasil dan Analisis
LSTM - Accuracy: 1.0, F1-score: 1.0

GRU - Accuracy: 1.0, F1-score: 1.0

Transformer - Accuracy: 1.0, F1-score: 1.0

### Antarmuka Pengguna
Aplikasi Flask menyediakan antarmuka sederhana dengan:
1. Form untuk memasukkan ulasan pelanggan.
2. Prediksi sentimen berdasarkan rating yang diproses oleh model.
3. Visualisasi hasil (positif, netral, negatif).

## File dalam Proyek
1. **preprocess.py**: Preprocessing teks ulasan dan menyimpan hasilnya.
2. **model.py**: Pelatihan model machine learning berbasis Random Forest.
3. **evaluasi.py**: Eksperimen dengan model deep learning (LSTM, GRU, Transformer).
4. **app.py**: Backend Flask untuk antarmuka pengguna.
5. **index.html**: Template HTML untuk form input.
6. **result.html**: Template HTML untuk menampilkan hasil prediksi.

## Cara Menjalankan Proyek
1. **Preprocessing Data**:
   ```bash
   python preprocess.py

2. **Pelatihan Model**:
   ```bash
   python model.py
   
3. **Preprocessing Data**:
   ```bash
   python preprocess.py

4. **Evaluasi Model**:
   ```bash
   python evaluasi.py
   
5. **Menjalankan Aplikasi Flask**:
   ```bash
   python app.py
