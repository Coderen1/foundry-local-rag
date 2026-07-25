# Yerel RAG Asistani (Foundry Local)

Microsoft Foundry Local ve sentence-transformers kullanarak tamamen cevrimdisi calisan,
dokuman tabanli bir soru-cevap (RAG — Retrieval-Augmented Generation) asistani.

> **Not:** Bu proje ilk tasarlandiginda embedding adiminin da Foundry Local uzerinden
> (`qwen3-embedding-0.6b` alias'i ile) yapilmasi planlanmisti. Ancak Foundry Local'in
> guncel model katalogunda (`foundry model list`) hicbir embedding modeli bulunmadigi
> tespit edildi — katalog yalnizca `chat-completion`, `vision-language-chat` ve
> `automatic-speech-recognition` gorevlerini icermektedir. Bu nedenle embedding adimi
> yerel olarak calisan `sentence-transformers` kutuphanesiyle yapilir; sohbet/uretim
> adimi ise Foundry Local (`phi-3.5-mini`) ile devam eder. Detay icin asagidaki
> "Tasarim Kararlari" bolumune bakin.

## Mimari

```
Kullanici Sorusu
      │
      ▼
  app.py (Streamlit arayuzu)
      │
      ▼
rag_engine.py ── sorguyu embed eder ──► sentence-transformers (yerel embedding modeli)
      │
      ├──► rag_store.db (SQLite) uzerinde cosine similarity ile en alakali parcalari bulur
      │
      ▼
Baglam + Sistem Prompt ──► Foundry Local (sohbet modeli, orn. Phi-3.5)
      │
      ▼
   Uretilen Yanit
```

Dosyalar:
- `ingest.py` — Dokumanlari (`.txt`, `.md`, `.pdf`) sayfa sayfa okur (gerekirse OCR ile),
  sayfa siniri icinde parcalara (chunk) boler, sentence-transformers ile embedding cikarir
  ve `rag_store.db` SQLite veritabanina (sayfa numaralariyla birlikte) kaydeder.
- `rag_engine.py` — Kullanici sorusunu embed eder, SQLite'taki parcalarla cosine similarity
  hesaplayarak en alakali baglami bulur ve Foundry Local sohbet modeline gonderip yanit alir;
  kaynaklari "dosya.pdf (sayfa 7)" seklinde sayfa numarasiyla gosterir.
- `app.py` — Dokuman yukleme/ice aktarma ve soru-cevap icin Streamlit web arayuzu.
- `evaluate.py` — `test_questions.json` icindeki cevaplanabilir/cevaplanamaz sorulari
  RAGEngine'e karsi calistirip gecti/kaldi raporu basan basit bir test scripti.
- `requirements.txt` — Gerekli Python kutuphaneleri.

## 1. On Kosullar

- Python 3.10 veya uzeri
- [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/get-started)
  kurulu olmali (Windows, macOS veya Linux desteklenir)
- **Tesseract OCR ve Poppler** (resim/diyagram agirlikli PDF sayfalarindan metin okuyabilmek
  icin — bkz. "OCR ile gorsel-agirlikli sayfalar" bolumu):

  **macOS:**
  ```bash
  brew install tesseract poppler
  ```

  **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt-get install tesseract-ocr poppler-utils
  ```

  **Windows:** [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki) ve
  [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) kurup
  her ikisinin de `bin` klasorlerini PATH'e ekleyin.

  > Bu adim atlanirsa uygulama yine calisir; sadece resim/diyagram agirlikli PDF sayfalarindaki
  > icerik eksik kalir ve ingest sirasinda bunun icin acik bir uyari gosterilir (bkz. Sorun
  > Giderme).

### Foundry Local kurulumu

**macOS:**
```bash
brew tap microsoft/foundrylocal
brew install foundrylocal
```

**Windows:**
```powershell
winget install Microsoft.FoundryLocal
```

Kurulumdan sonra hangi model alias'larinin makinenizde mevcut oldugunu kontrol edin:

```bash
foundry model list
```

Bu projede varsayilan olarak asagidaki modeller kullanilir:
- Sohbet modeli (Foundry Local): `phi-3.5-mini`
- Embedding modeli (sentence-transformers, Hugging Face Hub): `sentence-transformers/all-MiniLM-L6-v2`

Eger `foundry model list` ciktisinda farkli bir sohbet modeli alias'i goruyorsaniz veya
farkli bir embedding modeli kullanmak isterseniz, asagidaki ortam degiskenleriyle
degistirebilirsiniz (proje kok dizininde calistirin):

```bash
export RAG_CHAT_MODEL="phi-3.5-mini"
export RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

> Not: Ilk calistirmada hem Foundry Local (`FoundryLocalManager` uzerinden) hem de
> sentence-transformers ilgili modelleri otomatik olarak internetten indirir. Bu indirme
> islemleri tamamlandiktan sonra uygulama tamamen cevrimdisi calisir.

## 2. Kurulum

```bash
cd local-rag-project
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Dokumanlarinizi Ekleyin

Proje kok dizininde bir `docs/` klasoru olusturun ve soru-cevap yapmak istediginiz
`.txt`, `.md` veya `.pdf` dosyalarini bu klasore koyun:

```bash
mkdir -p docs
# ornek: docs/kurs_notlari.pdf, docs/sss.md, docs/kilavuz.txt
```

Alternatif olarak, uygulama arayuzundeki dosya yukleme alanini da kullanabilirsiniz (bkz. Adim 5).

## 4. Dokumanlari Ice Aktarma (Ingest) — Komut Satirindan

```bash
python ingest.py
```

Bu komut `docs/` klasorundeki tum dokumanlari okur, parcalara boler, her parca icin
embedding hesaplar ve `rag_store.db` SQLite dosyasina kaydeder. Mevcut verileri silmeden
eklemek isterseniz:

```bash
python ingest.py --no-clear
```

## 5. Uygulamayi Calistirma

```bash
streamlit run app.py
```

Tarayicinizda acilan arayuzden:
1. Sol menudeki dosya yukleme alanindan dokumanlarinizi yukleyip **"Yuklenen Dosyalari Kaydet"**
   butonuna basin (veya `docs/` klasorune elle dosya kopyalayin).
2. **"Ice Aktar (Ingest)"** butonuna basarak dokumanlarin islenmesini bekleyin. Ingest sirasinda
   tespit edilen kalite sorunlari (ör. bir sayfada OCR gerekti veya basarisiz oldu) sari uyari
   kutulariyla gosterilir.
3. Sidebar'daki **"top_k"** kaydiricisiyla her soru icin kac parca getirilecegini ayarlayabilirsiniz.
4. Alt kisimdaki sohbet kutusuna sorunuzu yazin ve yanitin uretilmesini bekleyin.
5. Her yanitin altindaki **"Kullanilan Kaynaklar"** bolumunden hangi dokuman ve sayfanin
   kullanildigini gorebilirsiniz (ör. `kilavuz.pdf (sayfa 7)`).

## 6. Test Etme

Orijinal proje planinin "Functional Testing" adimini otomatiklestiren `evaluate.py` scripti
ile, kendi dokumanlariniza gore hazirladiginiz cevaplanabilir/cevaplanamaz sorulari toplu
halde deneyebilirsiniz:

```bash
cp test_questions.example.json test_questions.json
# test_questions.json dosyasini kendi dokumanlariniza gore duzenleyin
python evaluate.py
```

Her soru icin `expect_answerable: true` ise cevapta gecmesini bekledginiz `expected_keywords`
kontrol edilir; `false` ise cevabin "bulunmuyor/bilmiyorum" gibi bir fallback ifadesi icermesi
beklenir. Script sonunda `X/Y test gecti` ozeti basar.

> **Ipucu:** `expected_keywords` secerken kucuk modelin sayilari bazen rakamla ("3") bazen
> yaziyla ("uc") yazabildigini unutmayin — testi gerektiginden kirilgan yapmamak icin her
> turlu ifade edilişte saglam kalacak bir kelime secin (ör. sayi yerine "grup").

## Tasarim Kararlari ve Sinirlamalar

- **OCR ile gorsel-agirlikli sayfalar:** PDF sayfalarinin cogu metin katmani icerir, ama slayt
  tabanli belgelerde bazi sayfalar tamamen diyagram/goruntu olabilir (ör. bir "Logic Gates"
  slaydinda 6 gomulu resme ragmen sadece 12 karakter duz metin cikabiliyor — bu proje
  gelistirilirken gercek bir dokumanda tespit edildi). `ingest.py`, gomulu resmi olan ve metin
  katmani `OCR_MIN_CHARS` (varsayilan 40 karakter) altinda kalan sayfalari otomatik olarak
  Tesseract ile OCR'dan gecirir. Tesseract/Poppler kurulu degilse veya OCR basarisiz olursa
  akis durmaz, sadece o sayfa icin acik bir uyari uretilir (bkz. Sorun Giderme).
- **Sayfa numarali kaynak gosterimi:** Chunk'lar sayfa siniri asilmadan olusturulur (her chunk
  tek bir PDF sayfasina karsilik gelir), bu sayede kaynaklar `dosya.pdf (sayfa N)` seklinde
  gosterilebilir. `.txt`/`.md` dosyalarinda sayfa kavrami olmadigi icin sadece dosya adi gosterilir.
- **Embedding icin Foundry Local yerine sentence-transformers:** Proje planinda embedding
  adiminin da Foundry Local uzerinden yapilmasi hedefleniyordu, ancak `foundry model list`
  ile yapilan kontrolde guncel katalogda hicbir embedding modeli bulunmadigi tespit edildi
  (yalnizca sohbet, gorsel-dil ve konusma tanima modelleri mevcut). Bu yuzden embedding
  `sentence-transformers/all-MiniLM-L6-v2` ile yerel olarak hesaplanir; Foundry Local yalnizca

- **Embedding icin Foundry Local yerine sentence-transformers:** Proje planinda embedding
  adiminin da Foundry Local uzerinden yapilmasi hedefleniyordu, ancak `foundry model list`
  ile yapilan kontrolde guncel katalogda hicbir embedding modeli bulunmadigi tespit edildi
  (yalnizca sohbet, gorsel-dil ve konusma tanima modelleri mevcut). Bu yuzden embedding
  `sentence-transformers/all-MiniLM-L6-v2` ile yerel olarak hesaplanir; Foundry Local yalnizca
  sohbet/uretim adiminda kullanilir. Ileride Foundry Local katalogunda embedding modeli
  eklenirse, `ingest.py` icindeki `get_embedding_model()`/`get_embedding()` fonksiyonlari
  Foundry Local'e geri tasinabilir.
- **Halusinasyon onleme:** Sistem prompt'u modelin yalnizca getirilen baglam icindeki
  bilgilere dayanarak cevap vermesini, aksi halde bilmedigini soylemesini zorunlu kilar
  (bkz. `rag_engine.py` icindeki `SYSTEM_PROMPT_TEMPLATE`).
- **Vektor arama:** Kucuk olcekli dokuman setleri icin SQLite'taki tum embedding'ler
  bellege okunup Python ile cosine similarity hesaplanir (brute-force). Cok buyuk dokuman
  setlerinde ozel bir vektor veritabani (orn. FAISS, Chroma) tercih edilmelidir.
- **Chunking:** Dokumanlar paragraf sinirlarina saygi gosterilerek ~1000 karakterlik
  parcalara bolunur; asiri uzun tek paragraflar 150 karakterlik ortusme (overlap) ile
  alt parcalara ayrilir. PDF'lerde chunk'lar sayfa sinirini asmaz (bkz. yukaridaki
  "Sayfa numarali kaynak gosterimi").
- **Model degisikligi:** Embedding modelini degistirirseniz (`RAG_EMBEDDING_MODEL`),
  mevcut vektorlerle karsilastirma anlamsiz hale gelecegi icin dokumanlari yeniden
  ice aktarmaniz (varsayilan olarak veritabanini temizleyen `ingest.py`) gerekir.

## Sorun Giderme

| Sorun | Cozum |
|---|---|
| `ModuleNotFoundError: No module named 'foundry_local'` | `foundry-local-sdk` 1.0.0'da paket adini (`foundry_local` -> `foundry_local_sdk`) ve tum API'yi degistirdi. Bu proje eski (tutorial uyumlu) API'yi kullanir; `pip show foundry-local-sdk` ile suruma bakin, 1.x kuruluysa `pip install "foundry-local-sdk>=0.3.0,<1.0.0"` ile 0.x surumune donun. Ayrica `streamlit`/`python` komutunuzun kullandigi ortamin (venv/conda) dogru oldugundan emin olun — `which streamlit` ile kontrol edebilirsiniz. |
| `Foundry is not installed or not on PATH` hatasi | Foundry Local CLI/servisi kurulu degil. Bu README'nin "Foundry Local kurulumu" bolumundeki `brew`/`winget` komutunu calistirip `foundry model list` ile dogrulayin. |
| `Embedding modeli yuklenemedi` hatasi | `sentence-transformers` modeli ilk kullanimda Hugging Face Hub'dan indirilir; internet baglantinizi kontrol edin. Kurumsal/firewall'lu aglarda Hugging Face erisimi engellenmis olabilir. |
| `Foundry Local ... modeli baslatilamadi` hatasi | Foundry Local servisinin calistigindan (`foundry service status`) ve model alias'inin `foundry model list` ciktisinda yer aldigindan emin olun. |
| Ice aktarma sirasinda `docs klasoru bos` hatasi | `docs/` klasorune en az bir `.txt`, `.md` veya `.pdf` dosyasi ekleyin. |
| Ingest sirasinda "metin katmani zayifti, OCR ile okundu" uyarisi | Bilgilendirme amaclidir, hata degildir — o sayfa resim/diyagram agirlikliydi ve OCR ile kurtarildi. |
| Ingest sirasinda "OCR calismadi" uyarisi | Tesseract/Poppler kurulu degil veya PATH'te degil. "On Kosullar" bolumundeki `brew`/`apt-get`/Windows kurulum adimini calistirip terminali yeniden baslatin. Bu uyari akisi durdurmaz; sadece o sayfanin icerigi eksik kalabilir. |
| Turkce dokumanlarda OCR sonucu bozuk cikiyor | Varsayilan OCR dili `eng`'dir. `brew install tesseract-lang` (veya Linux'ta `tesseract-ocr-tur` paketi) ile Turkce dil verisini kurup `export RAG_OCR_LANG="tur"` ayarlayin. |
| `Yanitlar cok yavas` | Daha kucuk bir sohbet modeli secin, veya sidebar'daki `top_k` kaydiricisini dusurun. |
| Port zaten kullaniliyor (Streamlit) | `streamlit run app.py --server.port 8502` gibi farkli bir port belirtin. |
