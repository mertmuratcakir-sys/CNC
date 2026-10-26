# cnc-simulator

Gradio uygulaması (Render'da çalışacak şekilde ipynb'den dönüştürüldü).

## Yerelde çalıştırma
```
pip install -r requirements.txt
python app.py
```

## Render'a yükleme
1. Bu klasörü GitHub'a yükle.
2. Render → New → Web Service → repoyu seç.
3. Build Command: `pip install -r requirements.txt`  |  Start Command: `python app.py`
   (veya New → Blueprint ile `render.yaml` kullan).
