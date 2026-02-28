[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

A drop-in subtitle generator built on OpenAI Whisper, extended with precise per-segment language detection and refinement for videos containing mixed languages.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

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

## 🗂 Project Structure

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

---

## 🔧 Installation

1. **Clone this repo**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
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

- Whisper model: `--whisper-model` (default `large`)
- Sampling rate: hard-coded to `16000` for processing
- FFmpeg extraction: mono WAV, `44100 Hz`, with `dynaudnorm=f=100`
- Lingua detector: initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow
- Allowed language codes for Whisper-side filtering include `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` in helper defaults

Assumption note: language lists in helper defaults and main detector setup are not fully identical; this README preserves current behavior as implemented.

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

---

## 🧭 Development Notes

- Canonical active script is `vad_lang_subtitle.py`.
- Historical files (`*.old`, `*.shorterlength*`, `archived/`) are useful for reference but appear non-canonical.
- There is currently no packaged project scaffolding (`pyproject.toml`, `setup.py`) and no CI/test suite committed.
- `data/` contains large sample media artifacts; be mindful of repository size and local disk usage during experiments.
- `clean_subtitles_dict()` exists in code but is currently not invoked by the main pipeline.

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

---

## 🗺 Roadmap

- Add and maintain a pinned `requirements.txt` or `pyproject.toml`.
- Add automated tests for segmentation and timestamp-cleanup logic.
- Add benchmark and quality evaluation docs for multilingual edge cases.
- Add optional config file support instead of code-default-only behavior.
- Expand i18n README set in `i18n/` and keep language bars synchronized.

---

## 💖 Support

If this project helps you, you can support development via:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

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

---

## 📄 License

MIT © Lachlan Chen
