[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Ein sofort einsatzbereiter Untertitel-Generator auf Basis von OpenAI Whisper, erweitert um präzise Sprach­erkennung und -verfeinerung pro Segment für Videos mit mehreren Sprachen.

> Erzeuge sauberere mehrsprachige Untertitel aus echtem, gemischtem Sprachmaterial mit sprachbewusster Segmentierung.

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

| Fokus | Wert |
| --- | --- |
| Input | FFmpeg-kompatibles Audio/Video |
| Pipeline | VAD-Segmentierung → Whisper-Transkription → Lingua-Feinschliff |
| Output | Normalisierte `*.wav`, `*.srt` und `*.json` |
| Beste Verwendung | Mehrsprachige Untertitel mit Sprach-Tags pro Segment |

---

## Inhaltsverzeichnis

- [Überblick](#-überblick)
- [Auf einen Blick](#auf-einen-blick)
- [Hauptmerkmale](#-hauptmerkmale)
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
- [Kontakt](#-kontakt)
- [Danksagungen](#-danksagungen)
- [Mitwirken](#-mitwirken)
- [Lizenz](#-lizenz)

---

## ✨ Überblick

`MultilingualWhisper` ist eine Python-CLI-Pipeline rund um [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Sie kombiniert:

- Silero VAD für Sprachsegmentierung
- OpenAI Whisper für Transkription und erste Spracherkennung
- Lingua für textbasierte Sprachverfeinerung
- FFmpeg für Extraktion, Normalisierung und Medienverarbeitung

Hauptergebnisse sind Untertiteldateien im Format `.srt` und `.json` sowie die extrahierte und normalisierte `.wav`-Audiodatei.

### Auf einen Blick

| Punkt | Details |
|---|---|
| Haupteinstieg | `vad_lang_subtitle.py` |
| Eingabe | FFmpeg-kompatibles Video/Audio |
| Ausgabe | `*.wav`, `*.srt`, `*.json` |
| Kernablauf | VAD -> Whisper -> Lingua -> Feinschliff |
| Typischer Anwendungsfall | Untertitel für gemischte Sprachen |

---

## 🚀 Hauptmerkmale

- **Silero VAD -> Whisper-Pipeline**
  Die Sprachaktivitätserkennung (VAD) teilt Audio in Sprachsegmente, und Whisper transkribiert anschließend jeden Teil.

- **Sprachdetektion in hoher Granularität**
  [Lingua](https://github.com/pemistahl/lingua-java) wird zusammen mit Whispers eigenem Detektor genutzt, um jedes Segment (sogar einzelne Wörter) mit ISO-Sprachcodes (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...) zu taggen.

- **Intelligente Segmentverfeinerung**
  Das Bereinigen der Zeitstempel sorgt dafür, dass keine Lücken oder Überlappungen entstehen. Interpunktionsteilung teilt lange Transkripte bei Kommas, Punkten, Fragezeichen usw. und ähnlichem auf. VAD ordnet Wörter anschließend wieder den VAD-Blöcken zu, damit die Untertitel flüssiger wirken. Längenabhängige Segmentierung nutzt sprachspezifische Grenzwerte.

- **Mehrsprachige Untertitel**
  Es werden `.srt` und `.json` ausgegeben, wobei Sprach-Tags pro Segment erhalten bleiben, damit du in nachgelagerten Playern oder Editoren später nach Sprache filtern oder stylen kannst.

- **Robuste Medienbehandlung**
  Audio wird automatisch via FFmpeg extrahiert und normalisiert, kaputte Container werden nach Möglichkeit repariert, und dynamische Normalisierung (`dynaudnorm`) sorgt für klarere Transkripte.

---

## 🔁 Pipeline-Ablauf

```text
Eingabemedium
  -> FFmpeg-Extraktion + Normalisierung (.wav)
  -> Silero VAD-Sprache-Zeitstempel
  -> Whisper-Transkription + Spracherkennung
  -> Lingua-Segment-Sprachverfeinerung
  -> Segment-Zusammenführung/-Aufteilung + Zeitstempelbereinigung
  -> Länge-sensibler Subtitle-Feinschliff
  -> Ausgabe .srt + .json
```

Hauptausführungsfluss in `vad_lang_subtitle.py`:

1. CLI-Argumente parsen (`--video-path`, `--whisper-model`, `--force`).
2. Ausgabepfade aus Basisname der Eingabe auflösen.
3. Audio via FFmpeg extrahieren/normalisieren.
4. Silero VAD (`torch.hub`) und Whisper-Modell laden.
5. Erste Transkription über VAD-Chunks.
6. Segmente zusammenführen/verfeinern, dann zweite Transkription über zusammengeführte Spannen.
7. Untertitel-Längenreduktion und Zeitstempelbereinigung anwenden.
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

> ⚠️ Hinweis: Das frühere README verwies auf `requirements.txt`, diese Datei fehlt jedoch aktuell im Repository-Root.

---

## ✅ Voraussetzungen

- Python `3.10+` (getestet mit aktuellen 3.x-Umgebungen)
- `ffmpeg` installiert und im `PATH`
- Ausreichend CPU/GPU + RAM für das gewählte Whisper-Modell (`large` empfiehlt sich stark mit GPU)
- Beim ersten Start Internetzugang, um Whisper-Modelldateien und Silero VAD-Assets (`torch.hub`) zu laden

Von dem Skript verwendete Python-Pakete:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python-Paket)
- `lingua-language-detector`
- `tqdm`

Schnell-Check-Befehle:

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

Wenn `requirements.txt` in deinem Checkout weiterhin fehlt, installiere die wichtigsten Laufzeitabhängigkeiten manuell:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Und stelle sicher, dass FFmpeg systemweit installiert ist.

---

## ⚡ Schnellstart

Wenn du den schnellsten Weg von `clone` zu Untertiteln willst:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Tipp: Nutze `small` in der Iteration, anschließend `large` für Endqualität.

Erwartete Artefakte neben der Eingabedatei:

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

| Flag | Alias | Pflicht | Beschreibung |
|---|---|---|---|
| `--video-path` | `-t` | Ja | Eingabemedium (Video/Audio, das FFmpeg unterstützt) |
| `--whisper-model` | — | Nein | Whisper-Modellname (Standard: `large`) |
| `--force` | — | Nein | Neu ausführen, auch wenn `.wav`, `.srt` oder `.json` bereits existieren |

### Verarbeitung

- Ausgabenamen werden aus dem Basispfad der Eingabe abgeleitet.
- Für `input.mp4` sind Ausgaben `input.wav` (normalisiertes Audio), `input.srt` (zeitgestempelte Untertitel) und `input.json` (Metadaten mit `start`, `end`, `lang`, `text`, optionalen Wortzeitstempeln).
- Vorhandene `.srt` oder `.json` führen zum Überspringen, außer `--force` ist gesetzt.

---

## ⚙️ Konfiguration

Aktuelle Konfiguration ist im Wesentlichen CLI-getrieben und in Code-Defaults festgelegt:

| Konfigurationsbereich | Verhalten |
|---|---|
| Whisper-Modell | `--whisper-model` (Standard `large`) |
| Verarbeitungssample-Rate | Fest auf `16000` für VAD/Transkription |
| FFmpeg-Extraktion | Mono-WAV, `44100 Hz`, mit `dynaudnorm=f=100` |
| Lingua-Detektor | Initialisiert für `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` im Hauptfluss |
| Whisper-side filtering defaults | Enthält `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Hinweis: Sprachlisten in den Hilfs-Defaults und in der Hauptdetektoreinstellung sind nicht vollständig identisch; dieses README gibt das aktuell implementierte Verhalten unverändert wieder.

---

## 📦 Ausgabeformat

Das Tool schreibt zwei Untertitel-Artefakte pro Eingabemedium:

- `*.srt`: Standard-Untertiteltext mit Zeitstempeln im Format `HH:MM:SS,mmm`.
- `*.json`: Strukturierte Untertitel-Liste mit formatierten Zeitstempeln und Sprach-Tags.

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

- `start`/`end` werden in JSON als SRT-konforme Strings serialisiert.
- `words` kann je nach Verarbeitungs-/Verfeinerungsstufe vorhanden sein.
- Ein `lang`-Wert von `und` kann bei unsicheren Sprachbereichen auftreten.

---

## 🧪 Beispiele

Ausführung auf einem MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Ausführung auf einer MOV-Datei mit Überschreiben:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Ausführung auf einer audio-only-Eingabe, die FFmpeg unterstützt:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Batch-Beispiel (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Entwicklungshinweise

- Die aktive kanonische Hauptdatei ist `vad_lang_subtitle.py`.
- Historische Dateien (`*.old`, `*.shorterlength*`, `archived/`) sind als Referenz nützlich, aber vermutlich nicht kanonisch.
- Es gibt aktuell kein Paket-Scaffolding (`pyproject.toml`, `setup.py`) und keine CI-/Test-Suite im Repo.
- `data/` enthält große Mediabeispiele/Ergebnisse; beachte Repo-Größe und lokalen Speicherverbrauch bei Experimenten.
- `clean_subtitles_dict()` existiert im Code, wird aber momentan nicht von der Hauptpipeline aufgerufen.
- `--force` ist der derzeitige Mechanismus, um Ausgaben bei iterativer Anpassung zuverlässig neu zu erzeugen.

Empfohlener lokaler Entwicklungs-Loop:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Verwende während der Iteration ein kleineres Modell (`tiny`/`base`/`small`) und wechsle danach auf `large` für finale Qualität.

---

## 🩺 Fehlerbehebung

| Symptom | Vorgehen |
|---|---|
| `ffmpeg: command not found` | FFmpeg installieren und mit `ffmpeg -version` prüfen. |
| Erster Start ist sehr langsam oder wirkt eingefroren | Erstmalige Modell-Downloads (Whisper + Silero) dauern; erneute Läufe sind schneller. |
| CUDA-/GPU-Fehler | Versuche CPU-Fallback mit einem kleineren Whisper-Modell (`small`, `base`, `tiny`) und prüfe einen passenden PyTorch-Build für die Umgebung. |
| Ausgabedateien werden nicht neu erstellt | Mit `--force` bestehende abgeleitete Dateien überschreiben. |
| `pip install -r requirements.txt` schlägt wegen fehlender Datei fehl | Nutze den im Installationsabschnitt genannten manuellen Befehl. |
| Ungenaue Sprachzuordnung bei kurzen Segmenten | Das kann bei sehr kurzen oder verrauschten Abschnitten passieren; die aktuelle Logik kombiniert Whisper und Lingua, hat aber weiterhin Randfälle. |
| Leere oder fast leere Untertitel-Ausgabe | Prüfe, ob die Eingabe gesprochene Segmente enthält, kontrolliere extrahiertes `.wav` und versuche anschließend erneut mit `--force` nach überprüfter FFmpeg-Extraktion. |
| Unerwartete Sprachwechsel zwischen benachbarten Zeilen | Das kann bei sehr kurzen Segmenten auftreten; konsolidiere nachträglich in nachgelagerten Tools nach Sprache und Mindestdauer. |

---

## ⚠️ Bekannte Einschränkungen und Annahmen

- Die Dependency-Manifest-Datei ist nicht committed (`requirements.txt`, `pyproject.toml` und `setup.py` fehlen derzeit im Repository-Root).
- In der README ist MIT als Lizenz angegeben, aber aktuell liegt keine eigenständige `LICENSE`-Datei vor.
- Lingua wird im Hauptfluss explizit mit `EN/ZH/JA/AR` initialisiert, während die Helper-Defaults mehr Kandidaten-Codes enthalten.
- Es sind keine automatischen Tests/Benchmarks committed, daher erfolgt Validierung primär manuell.
- Historische Skripte sind in Wurzel und `archived/` vorhanden; nur `vad_lang_subtitle.py` sollte als aktiv gelten, außer bewusstes Experimentieren.

---

## 🗺 Roadmap

- Gepinnte `requirements.txt` oder `pyproject.toml` pflegen und hinzufügen.
- Automatisierte Tests für Segmentierungs- und Zeitstempelbereinigungslogik ergänzen.
- Benchmark- und Qualitätsdokumentation für mehrsprachige Randfälle ergänzen.
- Optionale Konfigurationsdatei statt nur Code-Defaults einführen.
- i18n-README-Satz in `i18n/` ausbauen und Sprachleisten synchron halten.
- Sprachauswahlverhalten zwischen Detektorkonfiguration und Helper-Defaults klarer vereinheitlichen.
- Eine formale `LICENSE`-Datei ergänzen, um der README-Angabe zu entsprechen.

---

## 🔗 Danksagungen

- [OpenAI Whisper](https://github.com/openai/whisper) für Speech-to-Text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) für robuste Sprachaktivitätserkennung
- [Lingua](https://github.com/pemistahl/lingua-java) für hochpräzise Sprachekennung

---

## 🤝 Mitwirken

1. Forken und klonen
2. Neuen Branch erstellen: `git checkout -b feat/your-idea`
3. Committen und pushen
4. PR öffnen

Für größere Änderungen bitte folgendes beifügen:

- Kurze Beschreibung der erwarteten Verhaltensänderung
- Reproduzierbares Kommando-Beispiel
- Relevante Vorher-/Nachher-Untertitel-Snippets

## 📫 Kontakt

- Öffne ein Issue für Fehlerberichte, Nutzungsfragen und Feature-Requests.
- Nutze die Support-Optionen oben für Sponsoring- oder Spendenefragen.

## 📄 Lizenz

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
