[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

A drop-in subtitle generator built on OpenAI Whisper, extended with per-segment language detection and subtitle refinement for mixed-language media.

For a step-by-step walk-through of the pipeline and key functions, see [SCRIPT_LOGIC.md](SCRIPT_LOGIC.md).

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lingua](https://img.shields.io/badge/Language-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This repository also ships translated READMEs under [i18n/](i18n/). The English README is the source document; translated variants are generated from it and may lag behind by a commit.

---

## Overview

`vad_lang_subtitle.py` is a CLI pipeline that combines:

- Silero VAD for speech segmentation
- OpenAI Whisper for transcription and initial language prediction
- Lingua for text-based language refinement
- FFmpeg for media extraction, normalization, and repair

Primary outputs are:

- `*.wav` normalized extracted audio
- `*.srt` subtitles
- `*.json` structured subtitle metadata with language tags

Typical flow:

```text
input media
  -> FFmpeg extract + normalize
  -> Silero VAD timestamps
  -> Whisper transcription
  -> Lingua refinement
  -> merge/split/timestamp cleanup
  -> SRT + JSON
```

---

## Key Features

- Multilingual subtitle generation with per-segment language tags
- VAD-driven segmentation before transcription
- Timestamp cleanup to reduce overlaps and gaps
- Length-aware subtitle refinement for more readable output
- `--force` support to rebuild existing outputs
- Compatibility fallbacks for newer `torchaudio` and Whisper runtime changes

---

## Installation

1. Clone the repository.

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. Create and activate an environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing in your checkout, install the core runtime manually:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm soundfile
```

You also need `ffmpeg` available on `PATH`.

---

## Quick Start

```bash
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Use `small` while iterating, then rerun with `large` or `large-v3` for final output quality.

---

## Usage

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

CLI flags:

- `--video-path` / `-t`: input media path
- `--whisper-model`: Whisper model name
- `--force`: rebuild `.wav`, `.srt`, and `.json` even if they already exist

Outputs are written next to the input media by basename.

---

## Configuration Notes

Current runtime behavior is mostly code-default-driven:

- processing sample rate: `16000`
- FFmpeg extraction: mono WAV with dynamic normalization
- default Lingua detector set: English, Chinese, Japanese, Arabic
- VAD model source: `snakers4/silero-vad` via `torch.hub`

The current runtime also includes compatibility handling for:

- `torchaudio` releases where `list_audio_backends()` is unavailable
- `TorchCodec`-gated `torchaudio.load()` / `torchaudio.save()`
- Whisper word-timestamp alignment failures on newer Triton runtimes

---

## LazyEdit Integration

This repository is also used inside LazyEdit. LazyEdit should use the local copy of `whisper_with_lang_detect/vad_lang_subtitle.py`, not a separate checkout elsewhere.

Optional environment overrides used by LazyEdit:

```bash
LAZYEDIT_WHISPER_SCRIPT=/path/to/whisper_with_lang_detect/vad_lang_subtitle.py
LAZYEDIT_WHISPER_MODEL=large-v3
LAZYEDIT_WHISPER_FALLBACK_MODEL=large-v2
```

---

## Project Layout

```text
.
├── README.md
├── SCRIPT_LOGIC.md
├── vad_lang_subtitle.py
├── archived/
├── data/
├── figs/
└── i18n/
```

Notable historical variants such as `vad_lang_subtitle.py.old` and timestamped snapshots are preserved in the repo for reference.

---

## Troubleshooting

- If subtitle generation skips existing files, rerun with `--force`.
- If `torchaudio` load/save fails on newer releases, the current runtime falls back to `soundfile`.
- If Whisper word timestamps fail on the current GPU/Triton stack, the runtime retries without `word_timestamps`.
- If FFmpeg cannot decode a file, verify the input plays normally and that your local FFmpeg install is on `PATH`.

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- [Lingua](https://github.com/pemistahl/lingua-py)
- [FFmpeg](https://ffmpeg.org/)

---

## Contributing

1. Fork and clone the repo.
2. Create a focused branch.
3. Run the tool on a real sample when changing the transcription pipeline.
4. Open a PR with the runtime impact and sample command you validated.

---

## License

MIT © Lachlan Chen
