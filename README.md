# AVER: Audio-Visual Emotion Recognition

Rozpoznawanie emocji jednocześnie z głosu i mimiki twarzy w nagraniach wideo.
Projekt na przedmiot „Układy nieliniowe i uczenie maszynowe” (PW).

## Jak to działa

Z każdego nagrania .mp4 liczymy dwa embeddingi gotowymi, zamrożonymi modelami:

- audio: wav2vec2-base, uśredniony po czasie (768 liczb),
- wideo: 16 klatek, twarz wycinana detektorem Haar z OpenCV, ResNet-18, uśrednienie po klatkach (512 liczb).

Na każdym embeddingu osobno uczymy regresję logistyczną, a wynik końcowy to średnia prawdopodobieństw z obu modeli. Dane to RAVDESS (8 emocji). Zbiór testowy to osobni aktorzy, którzy nie występują w treningu.

## Instalacja

Python 3.11–3.14, Windows.

```
py -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

Paczki `Video_Speech_Actor_XX.zip` z https://zenodo.org/records/1188976 wypakować do `data/ravdess/`.

## Uruchomienie

```
python extract_features.py
python train.py
python predict.py sciezka/do/nagrania.mp4 --plot
```

`train.py` wypisuje accuracy oraz macro precision, recall i F1 dla audio, wideo i fuzji. `--plot` w `predict.py` zapisuje wykresy prawdopodobieństw z audio, wideo i fuzji jako `<nazwa nagrania>.png`.
