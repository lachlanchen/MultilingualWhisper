[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Ein sofort einsetzbarer Untertitel-Generator auf Basis von OpenAI Whisper, erweitert um präzise Sprachenerkennung und Verfeinerung pro Segment für Videos mit gemischten Sprachen.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Überblick

`MultilingualWhisper` ist eine Python-CLI-Pipeline mit Fokus auf [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Sie kombiniert:

- Silero VAD für Sprachsegmentierung
- OpenAI Whisper für Transkription und erste Sprachvorhersage
- Lingua für textbasierte Sprachverfeinerung
- FFmpeg für Extraktion, Normalisierung und Medienverarbeitung

Die primären Ausgaben sind Untertiteldateien in `.srt` und `.json` sowie extrahiertes, normalisiertes `.wav`-Audio.

### Auf einen Blick

| Element | Details |
|---|---|
| Haupteinstiegspunkt | `vad_lang_subtitle.py` |
| Eingabe | Von FFmpeg unterstütztes Video/Audio |
| Ausgabe | `*.wav`, `*.srt`, `*.json` |
| Kernablauf | VAD -> Whisper -> Lingua -> Verfeinerung |
| Typischer Anwendungsfall | Untertitelgenerierung für gemischte Sprachen |

---

## 🚀 Hauptfunktionen

- **Silero VAD -> Whisper-Pipeline**  
  Voice Activity Detection (VAD) teilt Audio in Sprachsegmente auf, danach transkribiert Whisper jeden Abschnitt.

- **Feingranulare Sprachenerkennung**  
  Verwendet [Lingua](https://github.com/pemistahl/lingua-java) zusammen mit Whispers eigener Erkennung, um jedes Segment (sogar einzelne Wörter) mit ISO-Sprachcodes zu markieren (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Intelligente Segment-Verfeinerung**  
  Zeitstempel-Bereinigung stellt sicher, dass es keine Lücken oder Überlappungen gibt. Interpunktions-Splits teilen lange Transkriptionen an Kommas, Punkten, Fragezeichen usw. VAD-Merges ordnen Wörter wieder VAD-Blöcken zu, um flüssigere Untertitel zu erzeugen. Längenabhängige Segmentierung verwendet sprachspezifische Grenzen.

- **Mehrsprachige Untertitel**  
  Gibt sowohl `.srt` als auch `.json` aus und erhält Sprach-Tags pro Segment, sodass du in nachgelagerten Playern oder Editoren nach Sprache gestalten oder filtern kannst.

- **Robuste Medienverarbeitung**  
  Extrahiert und normalisiert Audio automatisch über FFmpeg, versucht fehlerhafte Container zu reparieren und nutzt dynamische Normalisierung (`dynaudnorm`) für klarere Transkripte.

---

## 🗂 Projektstruktur

```text
.
├── README.md
├── vad_lang_subtitle.py               # Main pipeline: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old           # Legacy prototype
├── vad_lang_subtitle.py.20250706      # Historical snapshot
├── vad_lang_subtitle.py.shorterlength # Alternative historical variant
├── vad_lang_subtitle.py.shorterlength2# Alternative historical variant
├── vad_lang_subtitle.srt              # Example output
├── vad_lang_subtitle.json             # Example JSON
├── .github/
│   └── FUNDING.yml                    # Sponsor links
├── archived/                          # Old experiments/prototypes
├── data/                              # Optional sample media + generated outputs
├── figs/                              # Branding assets (banner/logo)
├── i18n/                              # Translation/readme workspace (currently present, empty)
└── .auto-readme-work/                 # README generation workspace artifacts
```

> ⚠️ Hinweis: In der vorherigen README wurde `requirements.txt` erwähnt, aber diese Datei fehlt derzeit im Repository-Root.

---

## ✅ Voraussetzungen

- Python `3.10+` (getestet mit aktuellen 3.x-Umgebungen)
- `ffmpeg` installiert und im `PATH` verfügbar
- Ausreichend CPU/GPU + RAM für das gewählte Whisper-Modell (für `large` wird GPU dringend empfohlen)
- Internetzugang beim ersten Lauf zum Laden der Whisper-Modellgewichte und Silero-VAD-Assets (`torch.hub`)

Vom Skript verwendete Python-Pakete:

- `torch`
- `torchaudio`
- `whisper` (OpenAI-Whisper-Python-Paket)
- `lingua-language-detector`
- `tqdm`

---

## 🔧 Installation

1. **Dieses Repository klonen**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **Virtuelle Umgebung erstellen und aktivieren**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

Falls `requirements.txt` in deinem Checkout weiterhin fehlt, installiere die Kern-Runtime-Abhängigkeiten manuell:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Stelle außerdem sicher, dass FFmpeg systemweit installiert ist.

---

## 🛠 Verwendung

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI-Optionen

| Flag | Alias | Erforderlich | Beschreibung |
|---|---|---|---|
| `--video-path` | `-t` | Ja | Eingabepfad für Medien (von FFmpeg unterstütztes Video/Audio) |
| `--whisper-model` | — | Nein | Whisper-Modellname (Standard: `large`) |
| `--force` | — | Nein | Erneut ausführen, auch wenn `.wav`, `.srt` oder `.json` bereits existieren |

### Verarbeitungsverhalten

- Ausgabenamen werden vom Basis-Pfad der Eingabe abgeleitet.
- Für `input.mp4` sind die Ausgaben `input.wav` (normalisiertes Audio), `input.srt` (zeitgestempelte Untertitel) und `input.json` (Metadaten inklusive `start`, `end`, `lang`, `text`, optional Wort-Zeitstempel).
- Vorhandene `.srt` oder `.json` führen zum Überspringen, außer `--force` ist gesetzt.

---

## ⚙️ Konfiguration

Die aktuelle Konfiguration ist hauptsächlich CLI-gesteuert und code-default-gesteuert:

- Whisper-Modell: `--whisper-model` (Standard `large`)
- Sampling-Rate: für die Verarbeitung fest auf `16000` codiert
- FFmpeg-Extraktion: mono WAV, `44100 Hz`, mit `dynaudnorm=f=100`
- Lingua-Detector: im Hauptablauf für `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` initialisiert
- Zulässige Sprachcodes für Whisper-seitige Filterung enthalten in den Helper-Standards `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`

Hinweis zur Annahme: Sprachlisten in den Helper-Standards und der Haupt-Detector-Konfiguration sind nicht vollständig identisch; diese README dokumentiert das aktuell implementierte Verhalten.

---

## 🧪 Beispiele

Auf einer MP4 ausführen:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Auf einer MOV ausführen und Überschreiben erzwingen:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Auf einer reinen Audioeingabe ausführen, die von FFmpeg unterstützt wird:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 Entwicklungshinweise

- Das kanonische aktive Skript ist `vad_lang_subtitle.py`.
- Historische Dateien (`*.old`, `*.shorterlength*`, `archived/`) sind als Referenz nützlich, wirken aber nicht-kanonisch.
- Es gibt derzeit kein paketiertes Projekt-Scaffolding (`pyproject.toml`, `setup.py`) und keine eingecheckte CI-/Test-Suite.
- `data/` enthält große Beispiel-Medienartefakte; achte bei Experimenten auf Repository-Größe und lokalen Speicherverbrauch.
- `clean_subtitles_dict()` existiert im Code, wird aber aktuell nicht von der Hauptpipeline aufgerufen.

---

## 🩺 Fehlerbehebung

| Symptom | Maßnahme |
|---|---|
| `ffmpeg: command not found` | FFmpeg installieren und mit `ffmpeg -version` prüfen. |
| Erster Lauf ist sehr langsam oder wirkt eingefroren | Erste Modelldownloads (Whisper + Silero) können dauern; weitere Läufe sind schneller. |
| CUDA-/GPU-Fehler | CPU-Fallback mit kleinerem Whisper-Modell (`small`, `base`, `tiny`) versuchen und auf passende PyTorch-Builds für die Umgebung achten. |
| Ausgabedateien werden nicht neu erzeugt | `--force` verwenden, um bestehende abgeleitete Dateien zu überschreiben. |
| `pip install -r requirements.txt` schlägt fehl, weil Datei nicht gefunden wurde | Manuelle Installationsanweisung aus dem Installationsabschnitt verwenden. |
| Ungenaue Sprach-Tags bei kurzen Segmenten | Kann bei sehr kurzen/rauschigen Abschnitten passieren; die aktuelle Logik kombiniert Whisper und Lingua, hat aber weiterhin Randfälle. |

---

## 🗺 Roadmap

- Gepinnte `requirements.txt` oder `pyproject.toml` ergänzen und pflegen.
- Automatisierte Tests für Segmentierungs- und Zeitstempel-Bereinigungslogik hinzufügen.
- Benchmark- und Qualitätsdokumentation für mehrsprachige Randfälle ergänzen.
- Optionale Konfigurationsdatei statt rein code-default-basierter Konfiguration hinzufügen.
- i18n-README-Satz in `i18n/` erweitern und Sprachleisten synchron halten.

---

## 💖 Unterstützung

Wenn dir dieses Projekt hilft, kannst du die Entwicklung hier unterstützen:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Persönliche Website: https://lazying.art
- Chat/Community: https://chat.lazying.art
- Ideen-/Projekt-Hub: https://onlyideas.art

---

## 🔗 Danksagungen

- [OpenAI Whisper](https://github.com/openai/whisper) für Speech-to-Text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) für robuste Voice Activity Detection
- [Lingua](https://github.com/pemistahl/lingua-java) für hochpräzise Sprachidentifikation

---

## 🤝 Mitwirken

1. Forken und klonen
2. Branch erstellen: `git checkout -b feat/your-idea`
3. Committen und pushen
4. PR öffnen

---

## 📄 Lizenz

MIT © Lachlan Chen
