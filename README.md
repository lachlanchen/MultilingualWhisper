<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

A drop‑in subtitle generator built on OpenAI Whisper, extended with precise per‑segment language detection and refinement—perfect for videos containing multiple languages.

For a step‑by‑step walk‑through of the pipeline and the key functions, see `SCRIPT_LOGIC.md`.

---

## 🚀 Key Features

- **Silero VAD** → Whisper pipeline  
  Voice Activity Detection (VAD) splits audio into speech segments, then Whisper transcribes each chunk.

- **Fine‑grained language detection**  
  Uses [Lingua](https://github.com/pemistahl/lingua‑java) alongside Whisper’s own detector to tag every segment (even individual words) with ISO language codes (en, zh, ja, ar, yue, ko, vi, es, fr, …).

- **Intelligent segment refinement**  
  - **Timestamp cleanup** ensures no gaps or overlaps  
  - **Punctuation splits** break long transcriptions at commas, periods, question marks, etc.  
  - **VAD merges** re‑align words back to VAD blocks for smoother subtitles  

- **Multilingual subtitles**  
  Outputs both `.srt` and `.json`, preserving language tags per segment so you can style or filter by language in downstream players or editors.

- **Robust video support**  
  Auto‑extracts & normalizes audio via FFmpeg, repairs corrupted containers, and normalizes volume for clearer transcripts.

---

## 🔧 Installation

1. **Clone this repo**  
   ```bash
   git clone git@github.com:lachlanchen/MultilingualWhisper.git
   cd MultilingualWhisper
   ```

2. **Create & activate** a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠 Usage

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

- `--video-path` (`-t`): input video file  
- `--whisper-model`: Whisper variant (tiny→large)  
- `--force`: re‑run even if `.wav`, `.srt`, or `.json` already exist  

After running you’ll get:

- `yourvideo.wav` (enhanced audio)  
- `yourvideo.srt` (timestamped subtitles)  
- `yourvideo.json` (rich JSON with `start`, `end`, `lang`, `text`, and word‑level timestamps)

---

## 🔌 LazyEdit Integration

This repo is also used as a submodule in LazyEdit. LazyEdit resolves the script path relative to the repo, so it runs the local copy at `whisper_with_lang_detect/vad_lang_subtitle.py`.

Optional LazyEdit env overrides:

```
LAZYEDIT_WHISPER_SCRIPT=/path/to/LazyEdit/whisper_with_lang_detect/vad_lang_subtitle.py
LAZYEDIT_WHISPER_MODEL=large-v3
LAZYEDIT_WHISPER_FALLBACK_MODEL=large-v2
```

---

## 📂 Project Layout

```
.
├── vad_lang_subtitle.py      # Main pipeline: VAD → Whisper → Lingua → refine → save
├── vad_lang_subtitle.py.old  # Legacy prototype
├── data/                     # Optional test media
├── archived/                 # Old experiments
├── vad_lang_subtitle.srt     # Example output
├── vad_lang_subtitle.json    # Example JSON
└── requirements.txt          # Python deps (whisper, torchaudio, lingua, silero‑vad, tqdm, etc.)
```

---

## 🔗 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech‑to‑text  
- [Snakers4/Silero‑VAD](https://github.com/snakers4/silero‑models) for robust voice activity detection  
- [Lingua](https://github.com/pemistahl/lingua‑java) for high‑accuracy language identification  

---

## 🤝 Contributing

1. Fork & clone  
2. Create a branch: `git checkout -b feat/your‑idea`  
3. Commit & push  
4. Open a PR—let’s make subtitles smarter!

---

## 📄 License

MIT © Lachlan Chen
