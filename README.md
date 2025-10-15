# Proyek Mini UTS Data Mining: Aplikasi Analisis Kesejahteraan Mahasiswa

**Mata Kuliah:** Data Mining (Ganjil 2025/2026)  
**Program Studi:** S-1 Teknik Informatika, Universitas Padjadjaran

---

### Deskripsi Proyek

Proyek ini bertujuan untuk mengaplikasikan konsep **Preprocessing** dan **Clustering** sebagai solusi inovatif berbasis data mining. Aplikasi ini dibangun untuk menganalisis data hasil survei mengenai tingkat kesejahteraan mahasiswa, mengelompokkannya ke dalam beberapa profil, dan menghasilkan rekomendasi kebijakan yang dapat ditindaklanjuti.

Sesuai dengan brief tugas, semua algoritma clustering utama (K-Means, DBSCAN) dan tahap preprocessing diimplementasikan secara manual (*from scratch*), hanya menggunakan library dasar seperti NumPy dan Pandas untuk manipulasi data. Aplikasi ini memiliki antarmuka pengguna interaktif yang dibangun menggunakan Streamlit.

### Daftar Isi
1. [Fitur Utama](#fitur-utama)
2. [Tampilan Aplikasi](#tampilan-aplikasi)
3. [Struktur Proyek](#struktur-proyek)
4. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
5. [Metodologi](#metodologi)
6. [Anggota Tim](#anggota-tim)
7. [Checklist Deliverables](#checklist-deliverables)

### Fitur Utama

* **Pipeline Preprocessing Lengkap**: Membersihkan data mentah, menangani nilai yang hilang, mendeteksi outlier, dan melakukan reduksi dimensi (PCA/UMAP).
* **Implementasi Algoritma Clustering *From Scratch***:
    * **K-Means**: Implementasi manual dengan inisialisasi K-Means++.
    * **DBSCAN**: Implementasi manual untuk menemukan cluster dengan bentuk non-konvensional dan mendeteksi noise.
* **Clustering Hirarkis**: Menggunakan `scipy.cluster` untuk analisis berbasis dendrogram.
* **Visualisasi Interaktif**: Antarmuka berbasis Streamlit yang memungkinkan pengguna untuk memilih algoritma, menyesuaikan parameter secara *real-time*, dan melihat hasilnya secara langsung.
* **Analisis Cluster Mendalam (Insight & Wisdom)**:
    * Secara otomatis membuat profil dan ringkasan kualitatif untuk setiap cluster (misal: "Akademik Sejahtera | Psikologis Rendah").
    * Menghasilkan rekomendasi kebijakan dan mengubahnya menjadi rencana aksi yang terstruktur, lengkap dengan KPI, penanggung jawab, dan estimasi biaya.
