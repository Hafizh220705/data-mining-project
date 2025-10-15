# Proyek Mini UTS Data Mining: Aplikasi Analisis Kesejahteraan Mahasiswa

**Mata Kuliah:** Data Mining (Ganjil 2025/2026)  
**Program Studi:** S-1 Teknik Informatika, Universitas Padjadjaran

---

### Deskripsi Proyek

Proyek ini bertujuan untuk mengaplikasikan konsep **Preprocessing** dan **Clustering** sebagai solusi inovatif berbasis data mining. Aplikasi ini dibangun untuk menganalisis data hasil survei mengenai tingkat kesejahteraan mahasiswa, mengelompokkannya ke dalam beberapa profil, dan menghasilkan rekomendasi kebijakan yang dapat ditindaklanjut.

Sesuai dengan brief tugas, semua algoritma clustering utama (K-Means, DBSCAN) dan tahap preprocessing diimplementasikan secara manual (*from scratch*), hanya menggunakan library dasar seperti NumPy dan Pandas untuk manipulasi data. Aplikasi ini memiliki antarmuka pengguna interaktif yang dibangun menggunakan Streamlit.

---

### Daftar Isi
1. [Fitur Utama](#fitur-utama)
2. [Tampilan Aplikasi](#tampilan-aplikasi)
3. [Struktur Proyek](#struktur-proyek)
4. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
5. [Metodologi](#metodologi)
6. [Anggota Tim](#anggota-tim)
7. [Checklist Deliverables](#checklist-deliverables)

---

### Fitur Utama

- **Pipeline Preprocessing Lengkap**  
  Alur kerja bertahap mulai dari mengunggah data, membersihkan (menangani nilai hilang & outlier), integrasi, transformasi (skor aspek & Z-score), reduksi dimensi (PCA/UMAP), hingga diskretisasi.
- **Implementasi Algoritma Clustering *From Scratch***  
  - **K-Means:** Implementasi manual dengan inisialisasi K-Means++.  
  - **DBSCAN:** Implementasi manual untuk menemukan cluster dengan bentuk non-konvensional dan mendeteksi noise.  
- **Clustering Hirarkis** menggunakan `scipy.cluster` untuk analisis berbasis dendrogram.  
- **Visualisasi Interaktif** dengan Streamlit untuk memilih algoritma, menyesuaikan parameter, dan menampilkan hasil secara *real-time*.  
- **Analisis Cluster Mendalam (Insight & Wisdom)**  
  - Membuat profil kualitatif tiap cluster (misal: *"Akademik Sejahtera | Psikologis Rendah"*).  
  - Menghasilkan rekomendasi kebijakan berupa rencana aksi dengan KPI, PIC, dan estimasi biaya yang dapat diekspor.

---

### Tampilan Aplikasi

![Screenshot Tampilan Utama Aplikasi](https://i.imgur.com/your-screenshot-1.png "Tampilan Utama Aplikasi")  
_Gambar 1: Halaman utama visualisasi clustering._

![Screenshot Hasil Analisis](https://i.imgur.com/your-screenshot-2.png "Hasil Analisis Cluster")  
_Gambar 2: Contoh tabel profil cluster dan rencana aksi yang dihasilkan._

---

### Struktur Proyek

Proyek ini diorganisir dengan struktur modular untuk memisahkan logika preprocessing, visualisasi, dan aplikasi utama.

```
├── Visualisasi/
│ ├── ClusterInsight.py # Analisis profil & rekomendasi
│ ├── ClusterWisdom.py # Rencana aksi berbasis insight
│ ├── DBSCAN_Visual.py # Implementasi manual DBSCAN & visualisasi
│ ├── Hierarchical_Visual.py # Implementasi Hierarchical Clustering
│ ├── KMeans_Visual.py # Implementasi manual K-Means & visualisasi
│ └── Visualisasi.py # Controller UI halaman visualisasi
│
├── app.py # Entry point aplikasi Streamlit
│
├── DataCleaning.py # Modul pembersihan data
├── DataDiscretization.py # Modul diskretisasi data
├── DataIntegration.py # Modul integrasi data
├── DataReduction.py # Modul reduksi dimensi (PCA/UMAP)
├── DataTransformation.py # Modul transformasi data kuesioner
│
├── requirements.txt # Daftar library
└── README.md # File dokumentasi ini
```
