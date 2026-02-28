[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

A drop-in subtitle generator built on OpenAI Whisper, extended with precise per-segment language detection and refinement for videos containing mixed languages.

> Generate cleaner multilingual subtitles from real-world mixed-language media with language-aware segmentation.

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

| Focus | Value |
| --- | --- |
| Input | FFmpeg-compatible audio/video |
| Pipeline | VAD segmentation → Whisper transcription → Lingua refinement |
| Output | Normalized `*.wav`, `*.srt`, and `*.json` |
| Best use | Mixed-language subtitles with per-segment language tags |

---

## Table of Contents

- [Overview](#-overview)
- [At a Glance](#at-a-glance)
- [Key Features](#-key-features)
- [Pipeline Flow](#-pipeline-flow)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Examples](#-examples)
- [Development Notes](#-development-notes)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations and Assumptions](#-known-limitations-and-assumptions)
- [Roadmap](#-roadmap)
- [Support](#-support)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Overview

`MultilingualWhisper` is a Python CLI pipeline centered on [`vad_lang_subtitle.py`](vad_lang_subtitle.py). It combines:

- Silero VAD for speech segmentation
- OpenAI Whisper for transcription and initial language prediction
- Lingua for text-based language refinement
- FFmpeg for extraction, normalization, and media handling

Primary outputs are subtitle files in `.srt` and `.json`, plus extracted normalized `.wav` audio.

### At a Glance

| Item | Details |
|---|---|
| Main entrypoint | `vad_lang_subtitle.py` |
| Input | Video/audio supported by FFmpeg |
| Output | `*.wav`, `*.srt`, `*.json` |
| Core flow | VAD -> Whisper -> Lingua -> refinement |
| Typical use case | Mixed-language subtitle generation |

---

## 🚀 Key Features

- **Silero VAD -> Whisper pipeline**
  Voice Activity Detection (VAD) splits audio into speech segments, then Whisper transcribes each chunk.

- **Fine-grained language detection**
  Uses [Lingua](https://github.com/pemistahl/lingua-java) alongside Whisper’s own detector to tag every segment (even individual words) with ISO language codes (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Intelligent segment refinement**
  Timestamp cleanup ensures no gaps or overlaps. Punctuation splits break long transcriptions at commas, periods, question marks, etc. VAD merges re-align words back to VAD blocks for smoother subtitles. Length-aware segmentation applies language-specific limits.

- **Multilingual subtitles**
  Outputs both `.srt` and `.json`, preserving language tags per segment so you can style or filter by language in downstream players or editors.

- **Robust media handling**
  Auto-extracts and normalizes audio via FFmpeg, attempts repair for broken containers, and applies dynamic normalization (`dynaudnorm`) for clearer transcripts.

---

## 🔁 Pipeline Flow

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

Main runtime path in `vad_lang_subtitle.py`:

1. Parse CLI args (`--video-path`, `--whisper-model`, `--force`).
2. Resolve output paths from input basename.
3. Extract/normalize audio via FFmpeg.
4. Load Silero VAD (`torch.hub`) and Whisper model.
5. First-pass transcription over VAD chunks.
6. Merge/refine segments, then second-pass transcription on merged spans.
7. Apply subtitle-length reduction and timestamp cleaning.
8. Save `.srt` and `.json`.

---

## 🗂 Project Structure

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

> ⚠️ Note: Previous README referenced `requirements.txt`, but it is currently missing in repository root.

---

## ✅ Prerequisites

- Python `3.10+` (tested with modern 3.x environments)
- `ffmpeg` installed and available on `PATH`
- Sufficient CPU/GPU + RAM for selected Whisper model (for `large`, GPU is strongly recommended)
- Internet access on first run to fetch Whisper model weights and Silero VAD assets (`torch.hub`)

Python packages used by the script include:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

Quick verification commands:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Installation

1. **Clone this repo**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is still absent in your checkout, install core runtime dependencies manually:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

And ensure FFmpeg is installed at system level.

---

## ⚡ Quick Start

If you want the fastest path from clone to subtitles:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Tip: use `small` while iterating, then switch to `large` for final quality.

Expected artifacts next to your input media:

- `*.wav` normalized extracted audio
- `*.srt` subtitle file for players/editors
- `*.json` structured multilingual subtitle metadata

---

## 🛠 Usage

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI Options

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### Processing Behavior

- Output names are derived from input base path.
- For `input.mp4`, outputs are `input.wav` (normalized audio), `input.srt` (timestamped subtitles), and `input.json` (metadata including `start`, `end`, `lang`, `text`, optionally word timings).
- Existing `.srt` or `.json` causes skip unless `--force` is set.

---

## ⚙️ Configuration

Current configuration is mainly CLI-driven and code-default-driven:

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Assumption note: language lists in helper defaults and main detector setup are not fully identical; this README preserves current behavior as implemented.

---

## 📦 Output Format

The tool writes two subtitle artifacts per input media:

- `*.srt`: Standard subtitle text with `HH:MM:SS,mmm` timestamps.
- `*.json`: Structured subtitle list containing formatted timestamps and language tags.

Typical JSON segment shape:

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

Notes:

- `start`/`end` are serialized as SRT-style strings in JSON output.
- `words` may be present depending on segment processing/refinement stage.
- A `lang` value of `und` can appear for uncertain language spans.

---

## 🧪 Examples

Run on an MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Run on a MOV and force overwrite:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Run on an audio-only input supported by FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Batch shell example (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Development Notes

- Canonical active script is `vad_lang_subtitle.py`.
- Historical files (`*.old`, `*.shorterlength*`, `archived/`) are useful for reference but appear non-canonical.
- There is currently no packaged project scaffolding (`pyproject.toml`, `setup.py`) and no CI/test suite committed.
- `data/` contains large sample media artifacts; be mindful of repository size and local disk usage during experiments.
- `clean_subtitles_dict()` exists in code but is currently not invoked by the main pipeline.
- `--force` is the current mechanism to guarantee regeneration of outputs for iterative tuning.

Suggested local dev loop:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Use a smaller model (`tiny`/`base`/`small`) while iterating, then switch to `large` for final output quality.

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | Install FFmpeg and verify with `ffmpeg -version`. |
| First run is very slow or appears stuck | Initial model downloads (Whisper + Silero) can take time; reruns are faster. |
| CUDA / GPU errors | Try CPU fallback by using a smaller Whisper model (`small`, `base`, `tiny`) and ensure matching PyTorch build for your environment. |
| Output files are not regenerated | Use `--force` to overwrite existing derived files. |
| `pip install -r requirements.txt` fails because file not found | Use manual dependency install command shown in Installation. |
| Inaccurate language tagging on short segments | This can happen on extremely short/noisy spans; current logic combines Whisper and Lingua but still has edge cases. |
| Empty or near-empty subtitle output | Confirm input has speech, inspect extracted `.wav`, and retry with `--force` after validating FFmpeg extraction. |
| Unexpected language flips between neighboring lines | This can occur on very short segments; consider post-merging in downstream tooling by language and minimum duration. |

---

## ⚠️ Known Limitations and Assumptions

- Dependency manifest is not committed (`requirements.txt`, `pyproject.toml`, and `setup.py` are absent in repository root at time of writing).
- License is declared in README as MIT, but a standalone `LICENSE` file is not currently present.
- Lingua is explicitly initialized with `EN/ZH/JA/AR` in main flow, while helper defaults include more candidate codes.
- No automated tests/benchmarks are currently committed, so validation is primarily manual.
- Historical scripts are present in root and `archived/`; only `vad_lang_subtitle.py` should be treated as active unless intentionally experimenting.

---

## 🗺 Roadmap

- Add and maintain a pinned `requirements.txt` or `pyproject.toml`.
- Add automated tests for segmentation and timestamp-cleanup logic.
- Add benchmark and quality evaluation docs for multilingual edge cases.
- Add optional config file support instead of code-default-only behavior.
- Expand i18n README set in `i18n/` and keep language bars synchronized.
- Clarify and unify language selection behavior between detector configuration and helper defaults.
- Add a formal `LICENSE` file to match README declaration.

---

## 🔗 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) for robust voice activity detection
- [Lingua](https://github.com/pemistahl/lingua-java) for high-accuracy language identification

---

## 🤝 Contributing

1. Fork and clone
2. Create a branch: `git checkout -b feat/your-idea`
3. Commit and push
4. Open a PR

For substantial changes, include:

- A short description of expected behavior change
- A reproducible command example
- Before/after subtitle snippets when relevant

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- Open an issue for bug reports, usage questions, and feature requests.
- Use the support options above for sponsorship and donation inquiries.

---

## 📄 License

MIT © Lachlan Chen
