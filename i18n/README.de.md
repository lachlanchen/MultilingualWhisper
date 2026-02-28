[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Ein direkt einsetzbarer Untertitel-Generator auf Basis von OpenAI Whisper, erweitert um präzise sprachspezifische Erkennung und Verfeinerung pro Segment für Videos mit gemischten Sprachen.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Inhaltsverzeichnis

- [Ubersicht](#-ubersicht)
- [Auf einen Blick](#auf-einen-blick)
- [Hauptfunktionen](#-hauptfunktionen)
- [Pipeline-Ablauf](#-pipeline-ablauf)
- [Projektstruktur](#-projektstruktur)
- [Voraussetzungen](#-voraussetzungen)
- [Installation](#-installation)
- [Verwendung](#-verwendung)
- [Konfiguration](#-konfiguration)
- [Ausgabeformat](#-ausgabeformat)
- [Beispiele](#-beispiele)
- [Entwicklungsnotizen](#-entwicklungsnotizen)
- [Fehlerbehebung](#-fehlerbehebung)
- [Bekannte Einschrankungen und Annahmen](#-bekannte-einschrankungen-und-annahmen)
- [Roadmap](#-roadmap)
- [Support](#-support)
- [Danksagungen](#-danksagungen)
- [Beitragen](#-beitragen)
- [Lizenz](#-lizenz)

---

## ✨ Ubersicht

`MultilingualWhisper` ist eine Python-CLI-Pipeline mit Fokus auf [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Sie kombiniert:

- Silero VAD fur Sprachsegmentierung
- OpenAI Whisper fur Transkription und erste Sprachvorhersage
- Lingua fur textbasierte Sprachverfeinerung
- FFmpeg fur Extraktion, Normalisierung und Medienverarbeitung

Die primaren Ausgaben sind Untertiteldateien in `.srt` und `.json` sowie extrahiertes, normalisiertes `.wav`-Audio.

### Auf einen Blick

| Punkt | Details |
|---|---|
| Haupteinstiegspunkt | `vad_lang_subtitle.py` |
| Eingabe | Von FFmpeg unterstutztes Video/Audio |
| Ausgabe | `*.wav`, `*.srt`, `*.json` |
| Kernablauf | VAD -> Whisper -> Lingua -> Verfeinerung |
| Typischer Anwendungsfall | Untertitelgenerierung fur gemischte Sprachen |

---

## 🚀 Hauptfunktionen

- **Silero VAD -> Whisper-Pipeline**  
  Voice Activity Detection (VAD) teilt Audio in Sprachsegmente auf, danach transkribiert Whisper jeden Abschnitt.

- **Feingranulare Spracherkennung**  
  Verwendet [Lingua](https://github.com/pemistahl/lingua-java) zusammen mit Whispers eigener Erkennung, um jedes Segment (sogar einzelne Worter) mit ISO-Sprachcodes zu kennzeichnen (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Intelligente Segmentverfeinerung**  
  Die Bereinigung von Zeitstempeln stellt sicher, dass keine Lucken oder Uberlappungen entstehen. Interpunktions-Splits teilen lange Transkriptionen an Kommata, Punkten, Fragezeichen usw. VAD-Merges richten Worter fur flussigere Untertitel wieder an VAD-Blocken aus. Langensensitive Segmentierung nutzt sprachspezifische Grenzen.

- **Mehrsprachige Untertitel**  
  Gibt sowohl `.srt` als auch `.json` aus und erhalt Sprach-Tags pro Segment, damit du in nachgelagerten Playern oder Editoren nach Sprache stylen oder filtern kannst.

- **Robuste Medienverarbeitung**  
  Extrahiert und normalisiert Audio automatisch uber FFmpeg, versucht defekte Container zu reparieren und nutzt dynamische Normalisierung (`dynaudnorm`) fur klarere Transkripte.

---

## 🔁 Pipeline-Ablauf

```text
Input media
  -> FFmpeg extract + normalize (.wav)
  -> Silero VAD speech timestamps
  -> Whisper transcription + language prediction
  -> Lingua segment language refinement
  -> Segment merge/split/timestamp cleanup
  -> Length-aware subtitle refinement
  -> Output .srt + .json
```

Hauptlaufzeitpfad in `vad_lang_subtitle.py`:

1. CLI-Argumente parsen (`--video-path`, `--whisper-model`, `--force`).
2. Ausgabepfade aus dem Eingabe-Basisnamen ableiten.
3. Audio uber FFmpeg extrahieren/normalisieren.
4. Silero VAD (`torch.hub`) und Whisper-Modell laden.
5. Erste Transkription uber VAD-Abschnitte.
6. Segmente zusammenfuhren/verfeinern, danach zweite Transkription auf zusammengefuhrten Bereichen.
7. Untertitellangen-Reduktion und Zeitstempel-Bereinigung anwenden.
8. `.srt` und `.json` speichern.

---

## 🗂 Projektstruktur

```text
.
├── README.md
├── vad_lang_subtitle.py                # Main pipeline: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old            # Legacy prototype
├── vad_lang_subtitle.py.20250706       # Historical snapshot
├── vad_lang_subtitle.py.shorterlength  # Alternative historical variant
├── vad_lang_subtitle.py.shorterlength2 # Alternative historical variant
├── vad_lang_subtitle.srt               # Example output
├── vad_lang_subtitle.json              # Example JSON
├── .github/
│   └── FUNDING.yml                     # Sponsor links
├── archived/
│   ├── vad.py
│   ├── vad_lang.py
│   ├── vad_lang_subtitle.py
│   ├── decode_audio.py
│   ├── decode_audio_v2.py
│   ├── text_language_detect.py
│   └── trans_with_lang.py
├── data/                               # Optional sample media + generated outputs
├── figs/                               # Branding assets (banner/logo)
├── i18n/                               # Existing multilingual README files
└── .auto-readme-work/                  # README generation workspace artifacts
```

> ⚠️ Hinweis: In einer fruheren README wurde `requirements.txt` referenziert, die Datei fehlt derzeit jedoch im Repository-Root.

---

## ✅ Voraussetzungen

- Python `3.10+` (getestet mit modernen 3.x-Umgebungen)
- `ffmpeg` installiert und auf `PATH` verfugbar
- Ausreichend CPU/GPU + RAM fur das gewahlte Whisper-Modell (fur `large` wird GPU dringend empfohlen)
- Internetzugang beim ersten Lauf zum Laden von Whisper-Modellgewichten und Silero-VAD-Assets (`torch.hub`)

Vom Skript verwendete Python-Pakete umfassen:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

Schnelle Verifikationsbefehle:

```bash
python --version
ffmpeg -version
```

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

3. **Abhangigkeiten installieren**

```bash
pip install -r requirements.txt
```

Falls `requirements.txt` in deinem Checkout weiterhin fehlt, installiere die zentralen Laufzeitabhangigkeiten manuell:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Und stelle sicher, dass FFmpeg auf Systemebene installiert ist.

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
| `--video-path` | `-t` | Ja | Eingabe-Medienpfad (von FFmpeg unterstutztes Video/Audio) |
| `--whisper-model` | — | Nein | Name des Whisper-Modells (Standard: `large`) |
| `--force` | — | Nein | Erneut ausfuhren, selbst wenn `.wav`, `.srt` oder `.json` bereits existieren |

### Verarbeitungsverhalten

- Ausgabenamen werden aus dem Eingabe-Basispfad abgeleitet.
- Fur `input.mp4` sind die Ausgaben `input.wav` (normalisiertes Audio), `input.srt` (Untertitel mit Zeitstempeln) und `input.json` (Metadaten inklusive `start`, `end`, `lang`, `text`, optional Wort-Timings).
- Vorhandene `.srt` oder `.json` fuhren zum Uberspringen, sofern `--force` nicht gesetzt ist.

---

## ⚙️ Konfiguration

Die aktuelle Konfiguration ist hauptsachlich CLI-gesteuert und code-default-gesteuert:

| Konfigurationsbereich | Aktuelles Verhalten |
|---|---|
| Whisper-Modell | `--whisper-model` (Standard `large`) |
| Verarbeitungs-Sample-Rate | Auf `16000` fur VAD/Transkriptionsverarbeitung fest kodiert |
| FFmpeg-Extraktion | Mono-WAV, `44100 Hz`, mit `dynaudnorm=f=100` |
| Lingua-Detektor | Im Hauptablauf fur `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` initialisiert |
| Standardwerte fur Whisper-seitige Filter-Helfer | Umfasst `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Hinweis zur Annahme: Sprachlisten in Helper-Standards und Haupt-Detektor-Setup sind nicht vollstandig identisch; diese README bewahrt das aktuell implementierte Verhalten.

---

## 📦 Ausgabeformat

Das Tool schreibt pro Eingabemedium zwei Untertitel-Artefakte:

- `*.srt`: Standard-Untertiteltext mit `HH:MM:SS,mmm`-Zeitstempeln.
- `*.json`: Strukturierte Untertitelliste mit formatierten Zeitstempeln und Sprach-Tags.

Typische JSON-Segmentstruktur:

```json
{
  "start": "00:00:01,234",
  "end": "00:00:03,456",
  "lang": "en",
  "text": "Hello world",
  "words": [
    {
      "word": " Hello",
      "start": 1234,
      "end": 1678,
      "probability": 0.98
    }
  ]
}
```

Hinweise:

- `start`/`end` werden in der JSON-Ausgabe als SRT-Strings serialisiert.
- `words` kann je nach Stufe der Segmentverarbeitung/-verfeinerung vorhanden sein.
- Ein `lang`-Wert von `und` kann bei unsicheren Sprachabschnitten auftreten.

---

## 🧪 Beispiele

Ausfuhrung mit einer MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Ausfuhrung mit einer MOV und erzwungenem Uberschreiben:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Ausfuhrung mit einer nur-Audio-Eingabe, die von FFmpeg unterstutzt wird:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Batch-Shell-Beispiel (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Entwicklungsnotizen

- Das kanonische aktive Skript ist `vad_lang_subtitle.py`.
- Historische Dateien (`*.old`, `*.shorterlength*`, `archived/`) sind als Referenz nutzlich, wirken aber nicht kanonisch.
- Derzeit gibt es kein paketiertes Projekt-Scaffolding (`pyproject.toml`, `setup.py`) und keine eingecheckte CI/Test-Suite.
- `data/` enthalt grosse Beispielmedien-Artefakte; beachte bei Experimenten Repository-Grosse und lokale Speichernutzung.
- `clean_subtitles_dict()` existiert im Code, wird aktuell aber nicht durch die Hauptpipeline aufgerufen.
- `--force` ist der aktuelle Mechanismus, um die Regenerierung von Ausgaben fur iteratives Tuning sicherzustellen.

Empfohlene lokale Entwicklungsrunde:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Nutze wahrend der Iteration ein kleineres Modell (`tiny`/`base`/`small`) und wechsle fur die finale Ausgabequalitat auf `large`.

---

## 🩺 Fehlerbehebung

| Symptom | Was zu tun ist |
|---|---|
| `ffmpeg: command not found` | FFmpeg installieren und mit `ffmpeg -version` verifizieren. |
| Der erste Lauf ist sehr langsam oder scheint zu hangen | Erste Modell-Downloads (Whisper + Silero) konnen Zeit benotigen; erneute Laufe sind schneller. |
| CUDA-/GPU-Fehler | CPU-Fallback mit kleinerem Whisper-Modell (`small`, `base`, `tiny`) versuchen und auf passendes PyTorch-Build fur deine Umgebung achten. |
| Ausgabedateien werden nicht neu erzeugt | `--force` verwenden, um bestehende abgeleitete Dateien zu uberschreiben. |
| `pip install -r requirements.txt` schlagt fehl, weil Datei nicht gefunden | Manuellen Installationsbefehl aus dem Abschnitt Installation verwenden. |
| Ungenaue Sprach-Tags bei kurzen Segmenten | Kann bei extrem kurzen/verrauschten Abschnitten vorkommen; aktuelle Logik kombiniert Whisper und Lingua, hat aber weiterhin Randfalle. |
| Leere oder nahezu leere Untertitel-Ausgabe | Prufen, ob die Eingabe Sprache enthalt, extrahiertes `.wav` inspizieren und nach validierter FFmpeg-Extraktion mit `--force` erneut versuchen. |
| Unerwartete Sprachwechsel zwischen benachbarten Zeilen | Kann bei sehr kurzen Segmenten auftreten; ggf. im Downstream-Tooling nach Sprache und Mindestdauer nachträglich mergen. |

---

## ⚠️ Bekannte Einschrankungen und Annahmen

- Ein Dependency-Manifest ist nicht eingecheckt (`requirements.txt`, `pyproject.toml` und `setup.py` fehlen zum Zeitpunkt des Schreibens im Repository-Root).
- Die Lizenz wird in der README als MIT angegeben, eine eigenstandige `LICENSE`-Datei ist derzeit jedoch nicht vorhanden.
- Lingua wird im Hauptablauf explizit mit `EN/ZH/JA/AR` initialisiert, wahrend Helper-Standards mehr Kandidatencodes enthalten.
- Es sind aktuell keine automatisierten Tests/Benchmarks eingecheckt, daher erfolgt die Validierung hauptsachlich manuell.
- Historische Skripte sind im Root und in `archived/` vorhanden; nur `vad_lang_subtitle.py` sollte als aktiv betrachtet werden, sofern nicht bewusst experimentiert wird.

---

## 🗺 Roadmap

- Eine gepinnte `requirements.txt` oder `pyproject.toml` hinzufugen und pflegen.
- Automatisierte Tests fur Segmentierung und Zeitstempel-Bereinigungslogik hinzufugen.
- Benchmark- und Qualitatsdokumentation fur mehrsprachige Randfalle hinzufugen.
- Optionale Konfigurationsdatei-Unterstutzung statt rein code-default-basierter Konfiguration hinzufugen.
- i18n-README-Set in `i18n/` erweitern und Sprachleisten synchron halten.
- Sprachwahlverhalten zwischen Detektor-Konfiguration und Helper-Standards klarstellen und vereinheitlichen.
- Eine formale `LICENSE`-Datei hinzufugen, passend zur README-Angabe.

---

## 💖 Support

Wenn dir dieses Projekt hilft, kannst du die Entwicklung unterstutzen uber:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personliche Website: https://lazying.art
- Chat/Community: https://chat.lazying.art
- Ideen-/Projekt-Hub: https://onlyideas.art

---

## 🔗 Danksagungen

- [OpenAI Whisper](https://github.com/openai/whisper) fur Speech-to-Text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) fur robuste Voice Activity Detection
- [Lingua](https://github.com/pemistahl/lingua-java) fur hochgenaue Sprachidentifikation

---

## 🤝 Beitragen

1. Forken und klonen
2. Branch erstellen: `git checkout -b feat/your-idea`
3. Committen und pushen
4. PR offnen

Fur umfangreichere Anderungen bitte beilegen:

- Eine kurze Beschreibung der erwarteten Verhaltensanderung
- Ein reproduzierbares Befehlsbeispiel
- Vorher-/Nachher-Untertitelausschnitte, falls relevant

---

## 📄 Lizenz

MIT © Lachlan Chen
