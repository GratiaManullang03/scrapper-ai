# 🤖 Scraper Cerdas AI

Scraper Cerdas AI adalah sebuah aplikasi web yang merevolusi cara ekstraksi data dari halaman web. Alih-alih menggunakan CSS selector atau XPath yang kaku, aplikasi ini memanfaatkan kecerdasan buatan (AI) untuk memahami instruksi dalam bahasa manusia dan mengekstrak data secara dinamis dari halaman web yang kompleks, bahkan yang dilindungi oleh sesi login.

Aplikasi ini menggunakan pendekatan hybrid yang canggih: AI digunakan untuk mengidentifikasi "peta" atau CSS selector menuju data, kemudian Playwright (sebuah browser automation tool) menggunakan peta tersebut untuk mengambil data dengan presisi 100%.

## ✨ Fitur Utama
- **Scraping pada Sesi Aktif**: Mengekstrak data dari halaman yang memerlukan login (dashboard internal, e-commerce, dll.) dengan "menumpang" pada sesi browser yang sudah aktif
- **Ekstraksi Berbasis Instruksi**: Cukup berikan perintah bahasa alami (contoh: "ekstrak semua nama produk dan harganya")
- **Pendekatan Hybrid (AI + Kode)**: Kombinasi pemahaman bahasa AI dengan presisi eksekusi Python
- **Penanganan `<iframe>`**: Mendeteksi dan masuk ke dalam `<iframe>` untuk ekstraksi data
- **Fleksibel**: Ekstraksi data tunggal (saldo) maupun berulang (riwayat transaksi)

## 🛠️ Teknologi yang Digunakan
- **Backend**: Python 3, Flask
- **Browser Automation**: Playwright
- **AI / LLM API**: Together.ai
- **Frontend**: HTML & CSS sederhana

## 🚀 Panduan Instalasi & Penggunaan

### Prasyarat
- Python 3.8+
- Google Chrome

### Langkah-langkah Instalasi
1. **Clone/Unduh Proyek**  
   Salin semua file proyek ke folder lokal.

2. **Buat Virtual Environment**  
   Buka terminal di folder proyek:
   ```bash
   # Buat virtual environment
   python -m venv venv

   # Aktifkan (Windows)
   venv\Scripts\activate

   # Aktifkan (macOS/Linux)
   source venv/bin/activate
   ```

3. **Install Dependensi**  
   Dengan venv aktif:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

4. **Jalankan Chrome Mode Debug**  
   Tutup semua Chrome, lalu buka dengan perintah:  

   **Windows**:
   ```cmd
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\ChromeProfileForScraping"
   ```

   **macOS**:
   ```bash
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="$HOME/ChromeProfileForScraping"
   ```

   **Linux**:
   ```bash
   google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/ChromeProfileForScraping"
   ```

5. **Atur API Key**  
   Buat file `.env` di folder proyek dengan konten:
   ```env
   TOGETHER_AI_API_KEY="MASUKKAN_API_KEY_ANDA_DI_SINI"
   ```

6. **Jalankan Aplikasi**  
   Di terminal dengan venv aktif:
   ```bash
   python -m flask run --port 5001
   ```

### Alur Kerja
1. Di jendela Chrome mode debug, login ke website target
2. Buka browser biasa, kunjungi `http://127.0.0.1:5001`
3. Salin URL target dari browser mode debug
4. Tempel URL dan tulis instruksi spesifik
5. Klik "Mulai Scrape" dan lihat hasilnya