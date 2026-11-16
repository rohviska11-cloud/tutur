# tutur. — panduan memperbarui website (model WavLM / HuBERT + SVM)

Website tetap statis di GitHub Pages, tanpa server. Model WavLM dan HuBERT dijalankan langsung
di browser pengunjung memakai **onnxruntime-web**, lalu hasilnya diklasifikasikan SVM.

| Skenario | Model | File di folder `model/` |
|---|---|---|
| 1 · Status bicara | WavLM-Base+ lapisan 8 → SVM-RBF (C=1) | `wavlm_l8.onnx.part000…`, `svm_speech_status.json/.bin` |
| 2 · Tingkat keparahan | HuBERT-Base lapisan 12 → SVM-RBF OvR (C=1) | `hubert_l12.onnx.part000…`, `svm_severity.json/.bin` |

## 1. Yang dijalankan di Colab

1. Buka `TORGO_DATASET_dengan_Bab_IX.ipynb`, jalankan sampai Bab IX selesai (embedding WavLM/HuBERT sudah ada di cache).
2. Tambah sel baru di paling bawah, tempel isi **`colab/sel_ekspor_web_ssl.py`**, lalu jalankan.
3. Perhatikan hasil cetakannya:
   - `cosine ONNX vs PyTorch` untuk setiap klip uji: sebaiknya **> 0,99**.
   - `Prediksi sama (embedding ONNX web vs embedding notebook)`: sebaiknya **≥ 95%**.
     Bila jauh di bawah itu, ubah `QUANTIZE = False` di atas sel lalu jalankan ulang (file jadi ±3× lebih besar).
   - `Total folder model`: perkiraan ±150–200 MB (dengan QUANTIZE = True).
4. Zip `tutur_web_model_ssl.zip` otomatis terunduh. Ekstrak: isinya folder `model/` dan `parity/`.

## 2. File yang diunggah ke GitHub

**Unggah / timpa:**

```
index.html
cek.html
PANDUAN.md                     (opsional)
assets/css/style.css
assets/js/app.js
assets/js/worker.js
assets/js/svm.js               (baru)
assets/js/dsp.js
assets/js/praat.js
assets/js/features.js
model/manifest.json            (dari zip Colab)
model/svm_speech_status.json   (dari zip Colab)
model/svm_speech_status.bin    (dari zip Colab)
model/svm_severity.json        (dari zip Colab)
model/svm_severity.bin         (dari zip Colab)
model/wavlm_l8.onnx.part000, part001, ...    (semua bagian)
model/hubert_l12.onnx.part000, part001, ...  (semua bagian)
```

**Hapus dari repo (model Random Forest lama, sudah tidak dipakai):**

```
assets/js/forest.js
model/speech_status.json
model/speech_status.bin
model/severity.json
model/severity.bin
```

Setiap bagian ONNX berukuran ≤ 19 MB, jadi bisa diunggah lewat **Add file → Upload files** di GitHub
(batas unggah lewat web 25 MB per file). Unggah beberapa kali bila perlu; GitHub membatasi 100 file per unggahan.

Folder `parity/` hanya untuk `cek.html`. Boleh diunggah sementara untuk pengecekan, lalu dihapus.

## 3. Coba di komputer sendiri (opsional, sebelum upload)

```bash
cd torgo-web
```

```bash
python -m http.server 8000
```

Buka `http://localhost:8000`. Klasifikasi pertama mengunduh model (tombol menampilkan persen unduhan),
setelah itu model tersimpan di cache browser sehingga klasifikasi berikutnya langsung jalan.

`http://localhost:8000/cek.html` membandingkan hasil browser dengan hasil Python untuk 12 klip dari `parity/`.
Kolom "status sama" dan "severity sama" seharusnya penuh, dengan selisih probabilitas sangat kecil.

## 4. Mengaktifkan GitHub Pages

Kalau repo sudah aktif sebelumnya, cukup unggah file di atas dan tunggu 1–2 menit.
Untuk repo baru: **Settings → Pages → Deploy from a branch → `main` / `(root)`**.

## 5. Hal yang perlu diketahui

- **Unduhan pertama besar** (±150–200 MB) karena WavLM dan HuBERT adalah model transformer. Setelah itu tersimpan di perangkat.
  Ukuran pastinya tertulis di bawah tombol Klasifikasikan.
- **Waktu proses** di laptop biasa sekitar 2–8 detik per rekaman 3 detik (CPU, satu thread). Di HP bisa lebih lama, dan HP dengan RAM kecil mungkin gagal memuat model.
- **Kartu "Ciri akustik"** (F0, HNR, jitter, shimmer) tetap dihitung dengan port Praat, tetapi hanya sebagai informasi. Model klasifikasi tidak memakainya.
- **Probabilitas** dihitung dengan Platt scaling (SVC `probability=True`). Label tetap mengikuti `decision_function`, sama seperti `predict` di notebook.
- Embedding di notebook dihitung di GPU dengan presisi campuran (fp16), sedangkan di website memakai ONNX int8. Perbedaannya kecil dan diukur oleh sel ekspor (baris "Prediksi sama").

## 6. Kalau ada masalah

| Gejala | Penyebab & solusi |
|---|---|
| Muncul "Model belum tersedia" | `model/manifest.json` belum diunggah |
| "gagal mengunduh …part00x (404)" | ada bagian ONNX yang belum diunggah; cek nama di `manifest.json` |
| Tombol berhenti di "Menyiapkan model..." lama sekali | perangkat kekurangan RAM; coba di laptop/PC |
| Ingin memaksa unduh ulang model | jalankan sel ekspor lagi (nilai `versi` di manifest berubah, cache lama otomatis tidak dipakai) |
| Hasil rekaman terasa beda dengan file TORGO | wajar: mikrofon, ruangan, dan bahasa berbeda dari data latih |
