[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Ein sofort einsetzbarer Untertitel-Generator auf Basis von OpenAI Whisper, erweitert um präzise Sprachenerkennung und -verfeinerung pro Segment für Videos mit gemischten Sprachen.

> Erzeuge sauberere mehrsprachige Untertitel aus realen Medien mit Sprachmischung durch sprachbewusste Segmentierung.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)
![Workflow](https://img.shields.io/badge/Flow-Silero%20%3E%20Whisper%20%3E%20Lingua-4D6D9A)
![Refinement](https://img.shields.io/badge/Refinement-Text%20%2B%20Timestamps-0EA5E9)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-6B7280)
![Maintained](https://img.shields.io/badge/Maintained-Yes-16A34A)

> 🌍 **Mehrsprachige Doku verfügbar**: Englisch + 10 übersetzte README-Varianten in [`i18n/`](i18n/), verlinkt in der Sprachleiste oben.

### Dokumentationssprachen

| Sprache | Datei |
| --- | --- |

| Fokus | Wert |
| --- | --- |
| Eingabe | FFmpeg-kompatibles Audio/Video |
| Pipeline | VAD-Segmentierung -> Whisper-Transkription -> Lingua-Verfeinerung |
| Ausgabe | Normalisierte `*.wav`, `*.srt` und `*.json` |
| Beste Verwendung | Mehrsprachige Untertitel mit Sprach-Tags pro Segment |

---

## Inhaltsverzeichnis

- [Überblick](#-überblick)
- [Auf einen Blick](#auf-einen-blick)
- [Wichtige Funktionen](#-wichtige-funktionen)
- [Pipeline-Ablauf](#-pipeline-ablauf)
- [Projektstruktur](#-projektstruktur)
- [Voraussetzungen](#-voraussetzungen)
- [Installation](#-installation)
- [Schnellstart](#-schnellstart)
- [Leitfaden zur Modellauswahl](#-leitfaden-zur-modellauswahl)
- [Verwendung](#-verwendung)
- [Konfiguration](#-konfiguration)
- [Ausgabeformat](#-ausgabeformat)
- [Beispiele](#-beispiele)
- [Entwicklungshinweise](#-entwicklungshinweise)
- [Fehlerbehebung](#-fehlerbehebung)
- [Bekannte Einschränkungen und Annahmen](#-bekannte-einschränkungen-und-annahmen)
- [Roadmap](#-roadmap)
- [Danksagungen](#-danksagungen)
- [Mitwirken](#-mitwirken)
- [Support](#-support)
- [Kontakt](#-kontakt)
- [Lizenz](#-lizenz)

---

## ✨ Überblick

`MultilingualWhisper` ist eine Python-CLI-Pipeline rund um [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Sie kombiniert:

- Silero VAD für Sprachsegmentierung
- OpenAI Whisper für Transkription und erste Sprachvorhersage
- Lingua für textbasierte Sprachverfeinerung
- FFmpeg für Extraktion, Normalisierung und Medienverarbeitung

Die primären Ausgaben sind Untertiteldateien in `.srt` und `.json` sowie extrahiertes, normalisiertes `.wav`-Audio.

### Auf einen Blick

| Punkt | Details |
|---|---|
| Haupteinstiegspunkt | `vad_lang_subtitle.py` |
| Eingabe | Von FFmpeg unterstütztes Video/Audio |
| Ausgabe | `*.wav`, `*.srt`, `*.json` |
| Kernablauf | VAD -> Whisper -> Lingua -> Verfeinerung |
| Typischer Anwendungsfall | Mehrsprachige Untertitel-Erzeugung |

---

## 🚀 Wichtige Funktionen

- **Silero VAD -> Whisper-Pipeline**  
  Voice Activity Detection (VAD) teilt Audio in Sprachsegmente, danach transkribiert Whisper jedes Segment.

- **Feingranulare Spracherkennung**  
  Nutzt [Lingua](https://github.com/pemistahl/lingua-java) zusammen mit Whispers eigener Erkennung, um jedes Segment (sogar einzelne Wörter) mit ISO-Sprachcodes zu markieren (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Intelligente Segmentverfeinerung**  
  Zeitstempel-Bereinigung verhindert Lücken und Überlappungen. Interpunktions-Splitting teilt lange Transkriptionen an Kommas, Punkten, Fragezeichen usw. VAD-Merges richten Wörter wieder an VAD-Blöcken aus, damit Untertitel flüssiger werden. Längenbewusste Segmentierung nutzt sprachspezifische Grenzwerte.

- **Mehrsprachige Untertitel**  
  Gibt sowohl `.srt` als auch `.json` aus und erhält Sprach-Tags pro Segment, sodass du in nachgelagerten Playern oder Editoren nach Sprache filtern oder gezielt stylen kannst.

- **Robuste Medienverarbeitung**  
  Extrahiert und normalisiert Audio automatisch via FFmpeg, versucht fehlerhafte Container zu reparieren und nutzt dynamische Normalisierung (`dynaudnorm`) für klarere Transkripte.

---

## 🔁 Pipeline-Ablauf

```text
Eingabemedium
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
2. Ausgabepfade aus dem Basenamen der Eingabe auflösen.
3. Audio via FFmpeg extrahieren/normalisieren.
4. Silero VAD (`torch.hub`) und Whisper-Modell laden.
5. Erste Transkriptionsrunde über VAD-Abschnitte.
6. Segmente zusammenführen/verfeinern, danach zweite Transkriptionsrunde auf zusammengeführten Spannen.
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

> ⚠️ Hinweis: In früheren README-Versionen wurde `requirements.txt` erwähnt, die Datei fehlt derzeit jedoch im Repository-Root.

---

## ✅ Voraussetzungen

- Python `3.10+` (getestet mit aktuellen 3.x-Umgebungen)
- Installiertes `ffmpeg`, verfügbar im `PATH`
- Ausreichend CPU/GPU + RAM für das gewählte Whisper-Modell (für `large` wird GPU dringend empfohlen)
- Internetzugang beim ersten Lauf zum Laden von Whisper-Gewichten und Silero-VAD-Assets (`torch.hub`)

Vom Skript genutzte Python-Pakete:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

Kurze Prüfkommandos:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Installation

1. **Repository klonen**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
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

Und stelle sicher, dass FFmpeg auf Systemebene installiert ist.

---

## ⚡ Schnellstart

Wenn du den schnellsten Weg von Clone bis zu Untertiteln willst:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Tipp: Nutze beim Iterieren `small`, wechsle für Endqualität anschließend zu `large`.

Erwartete Artefakte neben deinem Eingabemedium:

- `*.wav` normalisiertes, extrahiertes Audio
- `*.srt` Untertiteldatei für Player/Editoren
- `*.json` strukturierte mehrsprachige Untertitel-Metadaten

---

## 🎚 Leitfaden zur Modellauswahl

Wähle ein Whisper-Modell passend zu deinem Zielkonflikt aus Geschwindigkeit und Qualität:

| Modell | Geschwindigkeit | Qualität | Empfohlene Nutzung |
|---|---|---|---|
| `tiny` / `base` | Am schnellsten | Niedrigste | Schnelle Smoke-Tests und Pipeline-Validierung |
| `small` | Schnell | Gut | Tägliche Iteration und lokale Entwicklung |
| `medium` | Mittel | Besser | Ausgewogene Produktions-Workflows |
| `large` (default) | Am langsamsten | Beste | Finale Untertitel-Exporte mit höchster Qualität |

Praktisches Muster:

1. Mit `small --force` iterieren
2. Timing und Sprach-Tags validieren
3. Für das Liefer-Output mit `large --force` erneut ausführen

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
| `--video-path` | `-t` | Ja | Pfad zum Eingabemedium (Video/Audio von FFmpeg unterstützt) |
| `--whisper-model` | — | Nein | Whisper-Modellname (Standard: `large`) |
| `--force` | — | Nein | Erneut ausführen, auch wenn `.wav`, `.srt` oder `.json` bereits existieren |

### Verarbeitungsverhalten

- Ausgabenamen werden aus dem Basispfad der Eingabe abgeleitet.
- Für `input.mp4` sind die Ausgaben `input.wav` (normalisiertes Audio), `input.srt` (zeitgestempelte Untertitel) und `input.json` (Metadaten mit `start`, `end`, `lang`, `text`, optional Word-Timings).
- Existierende `.srt` oder `.json` führen zum Überspringen, außer `--force` ist gesetzt.

---

## ⚙️ Konfiguration

Die aktuelle Konfiguration ist hauptsächlich CLI-getrieben und durch Code-Defaults bestimmt:

| Konfigurationsbereich | Aktuelles Verhalten |
|---|---|
| Whisper-Modell | `--whisper-model` (default `large`) |
| Verarbeitungssamplerate | Fest auf `16000` für VAD/Transkriptionsverarbeitung |
| FFmpeg-Extraktion | Mono WAV, `44100 Hz`, mit `dynaudnorm=f=100` |
| Lingua-Detektor | Im Hauptablauf initialisiert für `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` |
| Whisper-side filtering helper defaults | Enthält `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Hinweis zur Annahme: Die Sprachlisten in Helper-Defaults und Hauptdetektor-Setup sind nicht vollständig identisch; dieses README spiegelt das aktuell implementierte Verhalten wider.

Zusätzliche Implementierungsdetails aus dem aktuellen Skript:

- `torch.set_num_threads(1)` wird zur Laufzeit gesetzt.
- Das VAD-Modell wird aus `snakers4/silero-vad` via `torch.hub.load(...)` geladen.
- Bei der Segmentbereinigung werden Einträge mit Sprache `und` oder leerem Text entfernt.

---

## 📦 Ausgabeformat

Das Tool schreibt pro Eingabemedium zwei Untertitel-Artefakte:

- `*.srt`: Standard-Untertiteltext mit Zeitstempeln im Format `HH:MM:SS,mmm`.
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

- `start`/`end` werden in der JSON-Ausgabe als SRT-ähnliche Strings serialisiert.
- `words` kann je nach Segmentverarbeitungs-/Verfeinerungsphase vorhanden sein.
- Ein `lang`-Wert `und` kann bei unsicheren Sprachabschnitten auftreten.

---

## 🧪 Beispiele

Ausführung mit einer MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Ausführung mit einer MOV und erzwungenem Überschreiben:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Ausführung mit einer reinen Audio-Eingabe, die FFmpeg unterstützt:

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

- Das kanonische aktive Skript ist `vad_lang_subtitle.py`.
- Historische Dateien (`*.old`, `*.shorterlength*`, `archived/`) sind als Referenz nützlich, wirken aber nicht kanonisch.
- Es gibt derzeit kein paketiertes Projekt-Scaffolding (`pyproject.toml`, `setup.py`) und keine commitete CI-/Test-Suite.
- `data/` enthält große Beispielmedien; achte bei Experimenten auf Repository-Größe und lokalen Speicherverbrauch.
- `clean_subtitles_dict()` existiert im Code, wird derzeit aber nicht in der Hauptpipeline verwendet.
- `--force` ist aktuell der Mechanismus, um bei iterativem Tuning die Ausgabe sicher neu zu erzeugen.

Empfohlener lokaler Entwicklungs-Loop:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Nutze beim Iterieren ein kleineres Modell (`tiny`/`base`/`small`) und wechsle für finale Ausgabequalität auf `large`.

---

## 🩺 Fehlerbehebung

| Symptom | Was tun |
|---|---|
| `ffmpeg: command not found` | FFmpeg installieren und mit `ffmpeg -version` prüfen. |
| Erster Lauf ist sehr langsam oder wirkt festgefahren | Die initialen Modell-Downloads (Whisper + Silero) können dauern; erneute Läufe sind schneller. |
| CUDA-/GPU-Fehler | CPU-Fallback mit kleinerem Whisper-Modell (`small`, `base`, `tiny`) testen und passenden PyTorch-Build für deine Umgebung sicherstellen. |
| Ausgabedateien werden nicht neu erzeugt | Mit `--force` bestehende abgeleitete Dateien überschreiben. |
| `pip install -r requirements.txt` schlägt fehl, weil die Datei nicht gefunden wird | Manuellen Abhängigkeits-Installationsbefehl aus dem Installationsabschnitt verwenden. |
| Ungenaue Sprach-Tags bei kurzen Segmenten | Das kann bei extrem kurzen/verrauschten Abschnitten passieren; die aktuelle Logik kombiniert Whisper und Lingua, hat aber weiterhin Randfälle. |
| Leere oder nahezu leere Untertitel-Ausgabe | Prüfen, ob Eingabe Sprache enthält, extrahierte `.wav` prüfen und nach validierter FFmpeg-Extraktion mit `--force` erneut testen. |
| Unerwartete Sprachwechsel zwischen benachbarten Zeilen | Das kann bei sehr kurzen Segmenten auftreten; ggf. in Downstream-Tools nach Sprache und Mindestdauer nachträglich zusammenführen. |
| FFmpeg-Extraktion schlägt bei beschädigten Medien fehl | Das Skript versucht einen erneuten Lauf nach Container-Reparatur (`-c copy -movflags +faststart`), stark beschädigte Dateien können trotzdem fehlschlagen. |

Schnelldiagnose:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Bekannte Einschränkungen und Annahmen

- Ein Abhängigkeits-Manifest ist nicht committed (`requirements.txt`, `pyproject.toml` und `setup.py` fehlen im Repository-Root zum Zeitpunkt des Schreibens).
- Die Lizenz ist in der README als MIT angegeben, eine eigenständige `LICENSE`-Datei ist derzeit jedoch nicht vorhanden.
- Lingua wird im Hauptablauf explizit mit `EN/ZH/JA/AR` initialisiert, während die Helper-Defaults mehr Kandidaten-Codes enthalten.
- Es sind derzeit keine automatisierten Tests/Benchmarks committed; die Validierung erfolgt primär manuell.
- Historische Skripte liegen im Root und in `archived/`; nur `vad_lang_subtitle.py` sollte als aktiv behandelt werden, außer bei bewusstem Experimentieren.
- Das Skript gibt aktuell ausführliche Laufzeit-Logs und Debug-Ausgaben pro Segment aus; dieses Verhalten ist in der aktuellen Implementierung erwartbar.

---

## 🗺 Roadmap

- Eine gepinnte `requirements.txt` oder `pyproject.toml` ergänzen und pflegen.
- Automatisierte Tests für Segmentierungs- und Zeitstempel-Bereinigungslogik ergänzen.
- Benchmark- und Qualitätsdokumentation für mehrsprachige Randfälle ergänzen.
- Optionale Config-Datei-Unterstützung statt ausschließlich Code-Defaults hinzufügen.
- Das i18n-README-Set in `i18n/` erweitern und Sprachleisten synchron halten.
- Sprachwahlverhalten zwischen Detektorkonfiguration und Helper-Defaults klarstellen und vereinheitlichen.
- Eine formale `LICENSE`-Datei hinzufügen, passend zur README-Angabe.

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

Bei größeren Änderungen bitte ergänzen:

- Eine kurze Beschreibung der erwarteten Verhaltensänderung
- Ein reproduzierbares Befehlsbeispiel
- Relevante Untertitel-Snippets vorher/nachher

---

## 📫 Kontakt

- Öffne ein Issue für Bug-Reports, Nutzungsfragen und Feature-Requests.
- Nutze die Support-Optionen oben für Sponsoring- und Spendenanfragen.

---

## 📄 Lizenz

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
