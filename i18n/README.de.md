[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Ein sofort einsetzbarer Untertitel-Generator auf Basis von OpenAI Whisper, erweitert um präzise Sprach­erkennung und Verfeinerung pro Segment für Videos mit gemischten Sprachen.

> Erzeuge sauberere mehrsprachige Untertitel aus realen Medien mit Sprachmischung durch sprachbewusste Segmentierung.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)

---

## Inhaltsverzeichnis

- [Überblick](#-überblick)
- [Auf einen Blick](#auf-einen-blick)
- [Hauptfunktionen](#-hauptfunktionen)
- [Pipeline-Ablauf](#-pipeline-ablauf)
- [Projektstruktur](#-projektstruktur)
- [Voraussetzungen](#-voraussetzungen)
- [Installation](#-installation)
- [Schnellstart](#-schnellstart)
- [Verwendung](#-verwendung)
- [Konfiguration](#-konfiguration)
- [Ausgabeformat](#-ausgabeformat)
- [Beispiele](#-beispiele)
- [Entwicklungshinweise](#-entwicklungshinweise)
- [Fehlerbehebung](#-fehlerbehebung)
- [Bekannte Einschränkungen und Annahmen](#-bekannte-einschränkungen-und-annahmen)
- [Roadmap](#-roadmap)
- [Support](#-support)
- [Danksagungen](#-danksagungen)
- [Mitwirken](#-mitwirken)
- [Lizenz](#-lizenz)

---

## ✨ Überblick

`MultilingualWhisper` ist eine Python-CLI-Pipeline mit [`vad_lang_subtitle.py`](vad_lang_subtitle.py) als Kern. Sie kombiniert:

- Silero VAD zur Sprachsegmentierung
- OpenAI Whisper für Transkription und erste Sprachvorhersage
- Lingua für textbasierte Sprachverfeinerung
- FFmpeg für Extraktion, Normalisierung und Medienverarbeitung

Die Hauptausgaben sind Untertiteldateien in `.srt` und `.json` sowie extrahiertes, normalisiertes `.wav`-Audio.

### Auf einen Blick

| Element | Details |
|---|---|
| Haupteinstiegspunkt | `vad_lang_subtitle.py` |
| Eingabe | Video/Audio, das von FFmpeg unterstützt wird |
| Ausgabe | `*.wav`, `*.srt`, `*.json` |
| Kernablauf | VAD -> Whisper -> Lingua -> Verfeinerung |
| Typischer Anwendungsfall | Erzeugung gemischtsprachiger Untertitel |

---

## 🚀 Hauptfunktionen

- **Silero VAD -> Whisper-Pipeline**
  Voice Activity Detection (VAD) teilt Audio in Sprachsegmente auf, danach transkribiert Whisper jeden Abschnitt.

- **Feingranulare Sprach­erkennung**
  Verwendet [Lingua](https://github.com/pemistahl/lingua-java) zusammen mit Whispers eigener Erkennung, um jedes Segment (sogar einzelne Wörter) mit ISO-Sprachcodes (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...) zu kennzeichnen.

- **Intelligente Segmentverfeinerung**
  Das Bereinigen von Zeitstempeln verhindert Lücken und Überlappungen. Interpunktions-Splits teilen lange Transkripte an Kommas, Punkten, Fragezeichen usw. VAD-Merges ordnen Wörter wieder den VAD-Blöcken zu, um flüssigere Untertitel zu erzeugen. Längenbewusste Segmentierung nutzt sprachspezifische Grenzen.

- **Mehrsprachige Untertitel**
  Gibt sowohl `.srt` als auch `.json` aus und behält Sprach-Tags pro Segment bei, sodass du nachgelagert in Playern oder Editoren nach Sprache stylen oder filtern kannst.

- **Robuste Medienverarbeitung**
  Extrahiert und normalisiert Audio automatisch per FFmpeg, versucht beschädigte Container zu reparieren und nutzt dynamische Normalisierung (`dynaudnorm`) für klarere Transkripte.

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
3. Audio über FFmpeg extrahieren/normalisieren.
4. Silero VAD (`torch.hub`) und Whisper-Modell laden.
5. Erste Transkription über VAD-Chunks.
6. Segmente zusammenführen/verfeinern, dann zweite Transkription auf zusammengeführten Bereichen.
7. Untertitel-Längenreduktion und Zeitstempel-Bereinigung anwenden.
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

> ⚠️ Hinweis: Das frühere README verwies auf `requirements.txt`, diese Datei fehlt aktuell jedoch im Repository-Root.

---

## ✅ Voraussetzungen

- Python `3.10+` (mit aktuellen 3.x-Umgebungen getestet)
- `ffmpeg` installiert und im `PATH` verfügbar
- Ausreichend CPU/GPU + RAM für das gewählte Whisper-Modell (für `large` wird GPU stark empfohlen)
- Internetzugang beim ersten Lauf, um Whisper-Modellgewichte und Silero-VAD-Assets (`torch.hub`) zu laden

Vom Skript verwendete Python-Pakete:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

Schnelle Verifizierungsbefehle:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Installation

1. **Dieses Repo klonen**

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

Falls `requirements.txt` in deinem Checkout weiterhin fehlt, installiere die Kernabhängigkeiten manuell:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Stelle außerdem sicher, dass FFmpeg auf Systemebene installiert ist.

---

## ⚡ Schnellstart

Wenn du den schnellsten Weg von Clone zu Untertiteln willst:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Tipp: Nutze beim Iterieren `small` und wechsle für die finale Qualität auf `large`.

Erwartete Artefakte neben deinem Eingabemedium:

- `*.wav` normalisiertes extrahiertes Audio
- `*.srt` Untertiteldatei für Player/Editoren
- `*.json` strukturierte mehrsprachige Untertitel-Metadaten

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
| `--video-path` | `-t` | Ja | Eingabemedienpfad (Video/Audio, das von FFmpeg unterstützt wird) |
| `--whisper-model` | — | Nein | Whisper-Modellname (Standard: `large`) |
| `--force` | — | Nein | Erneut ausführen, selbst wenn `.wav`, `.srt` oder `.json` bereits existieren |

### Verarbeitungsverhalten

- Ausgabenamen werden vom Eingabe-Basispfad abgeleitet.
- Für `input.mp4` sind die Ausgaben `input.wav` (normalisiertes Audio), `input.srt` (zeitkodierte Untertitel) und `input.json` (Metadaten inkl. `start`, `end`, `lang`, `text`, optional Wortzeiten).
- Vorhandene `.srt` oder `.json` führen zum Überspringen, außer `--force` ist gesetzt.

---

## ⚙️ Konfiguration

Die aktuelle Konfiguration wird hauptsächlich über CLI-Parameter und Code-Defaults gesteuert:

| Konfigurationsbereich | Aktuelles Verhalten |
|---|---|
| Whisper-Modell | `--whisper-model` (Standard `large`) |
| Verarbeitungs-Sample-Rate | Für VAD/Transkription fest auf `16000` gesetzt |
| FFmpeg-Extraktion | Mono WAV, `44100 Hz`, mit `dynaudnorm=f=100` |
| Lingua-Detektor | Im Hauptablauf für `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` initialisiert |
| Standardwerte für Whisper-seitige Filter-Helfer | Enthält `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Hinweis zur Annahme: Sprachlisten in den Helfer-Defaults und in der Hauptdetektor-Konfiguration sind nicht vollständig identisch; dieses README bildet das aktuell implementierte Verhalten ab.

---

## 📦 Ausgabeformat

Das Tool schreibt pro Eingabemedium zwei Untertitelartefakte:

- `*.srt`: Standard-Untertiteltext mit Zeitstempeln im Format `HH:MM:SS,mmm`.
- `*.json`: Strukturierte Untertitelliste mit formatierten Zeitstempeln und Sprach-Tags.

Typische Form eines JSON-Segments:

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

- `start`/`end` werden in der JSON-Ausgabe als SRT-ähnliche Strings serialisiert.
- `words` kann je nach Segment-Verarbeitungs-/Verfeinerungsphase vorhanden sein.
- Ein `lang`-Wert von `und` kann bei unsicheren Sprachabschnitten auftreten.

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

Batch-Shell-Beispiel (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Entwicklungshinweise

- Das kanonisch aktive Skript ist `vad_lang_subtitle.py`.
- Historische Dateien (`*.old`, `*.shorterlength*`, `archived/`) sind als Referenz nützlich, wirken aber nicht kanonisch.
- Aktuell gibt es kein paketiertes Projekt-Scaffolding (`pyproject.toml`, `setup.py`) und keine eingecheckte CI/Test-Suite.
- `data/` enthält große Beispielmedien-Artefakte; achte bei Experimenten auf Repository-Größe und lokalen Speicherverbrauch.
- `clean_subtitles_dict()` existiert im Code, wird in der Hauptpipeline aber aktuell nicht aufgerufen.
- `--force` ist derzeit der Mechanismus, um die Neuerzeugung von Ausgaben bei iterativer Abstimmung sicherzustellen.

Empfohlener lokaler Entwicklungszyklus:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Nutze beim Iterieren ein kleineres Modell (`tiny`/`base`/`small`) und wechsle für finale Ausgabequalität auf `large`.

---

## 🩺 Fehlerbehebung

| Symptom | Was zu tun ist |
|---|---|
| `ffmpeg: command not found` | FFmpeg installieren und mit `ffmpeg -version` prüfen. |
| Erster Lauf ist sehr langsam oder wirkt festgefahren | Die initialen Model-Downloads (Whisper + Silero) können dauern; erneute Läufe sind schneller. |
| CUDA-/GPU-Fehler | CPU-Fallback mit kleinerem Whisper-Modell (`small`, `base`, `tiny`) versuchen und auf passenden PyTorch-Build für deine Umgebung achten. |
| Ausgabedateien werden nicht neu erzeugt | `--force` verwenden, um bestehende abgeleitete Dateien zu überschreiben. |
| `pip install -r requirements.txt` schlägt fehl, weil Datei fehlt | Manuellen Installationsbefehl aus dem Installationsabschnitt verwenden. |
| Ungenaue Sprach-Tags bei kurzen Segmenten | Kann bei extrem kurzen/verrauschten Abschnitten passieren; die aktuelle Logik kombiniert Whisper und Lingua, hat aber weiterhin Randfälle. |
| Leere oder nahezu leere Untertitelausgabe | Prüfen, ob die Eingabe Sprache enthält, extrahiertes `.wav` kontrollieren und nach validierter FFmpeg-Extraktion mit `--force` erneut ausführen. |
| Unerwartete Sprachwechsel zwischen benachbarten Zeilen | Kann bei sehr kurzen Segmenten auftreten; ggf. nachgelagert nach Sprache und Mindestdauer zusammenführen. |

---

## ⚠️ Bekannte Einschränkungen und Annahmen

- Abhängigkeitsmanifest ist nicht eingecheckt (`requirements.txt`, `pyproject.toml` und `setup.py` fehlen im Repository-Root zum Zeitpunkt der Erstellung).
- Die Lizenz wird im README als MIT angegeben, eine eigenständige `LICENSE`-Datei ist derzeit jedoch nicht vorhanden.
- Lingua wird im Hauptablauf explizit mit `EN/ZH/JA/AR` initialisiert, während Helfer-Defaults mehr Kandidatencodes enthalten.
- Es sind derzeit keine automatisierten Tests/Benchmarks eingecheckt; Validierung ist primär manuell.
- Historische Skripte liegen im Root und in `archived/`; nur `vad_lang_subtitle.py` sollte als aktiv gelten, außer du experimentierst absichtlich.

---

## 🗺 Roadmap

- Eine versionierte `requirements.txt` oder `pyproject.toml` hinzufügen und pflegen.
- Automatisierte Tests für Segmentierung und Zeitstempel-Bereinigungslogik ergänzen.
- Benchmark- und Qualitätsbewertungsdokumente für mehrsprachige Randfälle ergänzen.
- Optionalen Config-Datei-Support hinzufügen statt ausschließlich Code-Defaults.
- i18n-README-Set in `i18n/` erweitern und Sprachleisten synchron halten.
- Sprachauswahlverhalten zwischen Detektor-Konfiguration und Helfer-Defaults klären und vereinheitlichen.
- Formale `LICENSE`-Datei ergänzen, passend zur README-Angabe.

---

## ❤️ Support

Wenn dir dieses Projekt Zeit spart, helfen Beiträge bei Wartung und zukünftigen Verbesserungen.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

Zusätzliche Support-/Community-Links:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

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

Für größere Änderungen bitte Folgendes beilegen:

- Eine kurze Beschreibung der erwarteten Verhaltensänderung
- Ein reproduzierbares Befehlsbeispiel
- Vorher/Nachher-Untertitel-Snippets, wenn relevant

---

## 📄 Lizenz

MIT © Lachlan Chen
