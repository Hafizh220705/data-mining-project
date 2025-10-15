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
2. [Struktur Proyek](#struktur-proyek)
3. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
4. [Metodologi](#metodologi)
5. [Anggota Tim](#anggota-tim)

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

---

### Instalasi & Cara Menjalankan

**1. Clone Repositori**
```bash
git clone https://github.com/Hafizh220705/data-mining-project.git
cd data-mining-project
```
**2. Buat dan Aktifkan Virtual Environment**
```bash
# Buat environment
python -m venv venv

# Aktifkan di Windows
venv\Scripts\activate

# Aktifkan di macOS/Linux
source venv/bin/activate
```
**3. Instal Dependensi**
```bash
pip install -r requirements.txt
```

**4. Siapkan Dataset**
Unduh data hasil survei dari Google Sheets dalam format .xlsx dan siapkan untuk diunggah melalui antarmuka aplikasi.

**5. Jalankan Aplikasi**
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser Anda.

---
### Metodologi
Proyek ini mengikuti metodologi CRISP-DM yang mencakup:
1. Business & Data Understanding – Merumuskan masalah kesejahteraan mahasiswa dan mengumpulkan data survei.
2. Data Preparation – Melakukan cleaning, integrasi, dan transformasi pada tahap awal aplikasi.
3. Modeling – Menerapkan K-Means, DBSCAN, dan Hierarchical Clustering interaktif di Streamlit.
4. Evaluation – Menganalisis hasil dengan metrik seperti Silhouette Score dan interpretasi profil.
5. Deployment – Menyajikan hasil analisis dalam aplikasi Streamlit lengkap dengan insight & rencana aksi.

---
### Anggota Tim

| No | Nama Lengkap              | NPM          |
|----|---------------------------|--------------|
| 1  | David Christian Nathaniel | 140810230027 | 
| 2  | Dzacky Ahmad              | 140810230043 |
| 3  | Hafizh Fadhl Muhammad     | 140810230070 | 
| 4  | Farhan Zia Rizky          | 140810230074 |
| 5  | Gideon Tamba              | 140810230082 |
