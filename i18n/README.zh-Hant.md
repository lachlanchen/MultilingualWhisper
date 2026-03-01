[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

一個以 OpenAI Whisper 為核心的即插即用字幕產生器，進一步強化為可對混合語言影片做精準的逐片段語言偵測與優化。

> 以語言感知分段，從真實世界的混合語言媒體產生更乾淨的多語字幕。

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


### 文件語言

| Locale | File |
| --- | --- |

| Focus | Value |
| --- | --- |
| Input | FFmpeg 相容音訊/影片 |
| Pipeline | VAD segmentation -> Whisper transcription -> Lingua refinement |
| Output | 正規化 `*.wav`、`*.srt` 與 `*.json` |
| Best use | 逐片段語言標記的混合語言字幕 |

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
- [Model Selection Guide](#-model-selection-guide)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Examples](#-examples)
- [Development Notes](#-development-notes)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations and Assumptions](#-known-limitations-and-assumptions)
- [Roadmap](#-roadmap)
- [Acknowledgments](#-acknowledgments)
- [Contributing](#-contributing)
- [Support](#-support)
- [Contact](#-contact)
- [License](#-license)

---

## ✨ Overview

`MultilingualWhisper` 是一套以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 為核心的 Python CLI 流程，整合了：

- 使用 Silero VAD 進行語音分段
- 使用 OpenAI Whisper 進行轉錄與初步語言預測
- 使用 Lingua 進行文本層語言修正
- 使用 FFmpeg 進行抽取、正規化與媒體處理

主要輸出為 `.srt` 與 `.json` 字幕檔，以及抽出的正規化 `.wav` 音訊。

### At a Glance

| Item | Details |
|---|---|
| Main entrypoint | `vad_lang_subtitle.py` |
| Input | FFmpeg 支援的影片/音訊 |
| Output | `*.wav`, `*.srt`, `*.json` |
| Core flow | VAD -> Whisper -> Lingua -> refinement |
| Typical use case | 混合語言字幕生成 |

---

## 🚀 Key Features

- **Silero VAD -> Whisper pipeline**  
  先以語音活動偵測（VAD）將音訊切成語音片段，再由 Whisper 逐段轉錄。

- **Fine-grained language detection**  
  結合 [Lingua](https://github.com/pemistahl/lingua-java) 與 Whisper 內建偵測器，對每個片段（甚至單字）標註 ISO 語言代碼（`en`、`zh`、`ja`、`ar`、`yue`、`ko`、`vi`、`es`、`fr` 等）。

- **Intelligent segment refinement**  
  時間戳清理可避免空隙與重疊。標點切分可在逗號、句號、問號等位置拆分長轉錄。VAD 合併會把詞重新對齊到 VAD 區塊，讓字幕更流暢。長度感知分段會套用語言特定限制。

- **Multilingual subtitles**  
  同時輸出 `.srt` 與 `.json`，並保留每段語言標記，方便你在後續播放器或編輯器中依語言樣式化或篩選。

- **Robust media handling**  
  透過 FFmpeg 自動抽取與正規化音訊，會嘗試修復損壞容器，並套用動態正規化（`dynaudnorm`）以提升轉錄清晰度。

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

`vad_lang_subtitle.py` 的主要執行路徑：

1. 解析 CLI 參數（`--video-path`、`--whisper-model`、`--force`）。
2. 從輸入檔 basename 推導輸出路徑。
3. 透過 FFmpeg 抽取並正規化音訊。
4. 載入 Silero VAD（`torch.hub`）與 Whisper 模型。
5. 對 VAD 分段進行第一輪轉錄。
6. 合併/優化片段後，對合併區段進行第二輪轉錄。
7. 套用字幕長度縮減與時間戳清理。
8. 儲存 `.srt` 與 `.json`。

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

> ⚠️ 注意：先前 README 提到 `requirements.txt`，但目前在儲存庫根目錄中不存在。

---

## ✅ Prerequisites

- Python `3.10+`（已在現代 3.x 環境測試）
- 已安裝 `ffmpeg`，且可在 `PATH` 中找到
- 依所選 Whisper 模型具備足夠 CPU/GPU 與 RAM（若用 `large`，強烈建議 GPU）
- 首次執行需可連網下載 Whisper 權重與 Silero VAD 資產（`torch.hub`）

腳本使用的 Python 套件包含：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python 套件）
- `lingua-language-detector`
- `tqdm`

快速檢查指令：

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

若你的版本仍缺少 `requirements.txt`，請手動安裝核心執行相依：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

並確認系統層級已安裝 FFmpeg。

---

## ⚡ Quick Start

若你想從 clone 直接快速產生字幕：

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

提示：迭代階段可先用 `small`，最終品質再改用 `large`。

在輸入媒體旁預期會產生：

- `*.wav` 正規化抽取音訊
- `*.srt` 供播放器/編輯器使用的字幕檔
- `*.json` 結構化多語字幕中繼資料

---

## 🎚 Model Selection Guide

依速度與品質需求選擇 Whisper 模型：

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | Fastest | Lowest | 快速 smoke test 與流程驗證 |
| `small` | Fast | Good | 日常迭代與本機開發 |
| `medium` | Medium | Better | 兼顧效率與品質的生產流程 |
| `large` (default) | Slowest | Best | 最終高品質字幕輸出 |

實務建議流程：

1. 先用 `small --force` 迭代
2. 驗證時間軸與語言標記
3. 最後改用 `large --force` 產出交付版本

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
| `--video-path` | `-t` | Yes | 輸入媒體路徑（FFmpeg 支援的音訊/影片） |
| `--whisper-model` | — | No | Whisper 模型名稱（預設：`large`） |
| `--force` | — | No | 即使 `.wav`、`.srt` 或 `.json` 已存在也強制重跑 |

### Processing Behavior

- 輸出檔名由輸入檔基底路徑推導。
- 對 `input.mp4`，輸出為 `input.wav`（正規化音訊）、`input.srt`（含時間戳字幕）、`input.json`（含 `start`、`end`、`lang`、`text`，可選詞級時間戳中繼資料）。
- 若 `.srt` 或 `.json` 已存在會跳過，除非設定 `--force`。

---

## ⚙️ Configuration

目前設定主要由 CLI 參數與程式內預設值控制：

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model`（預設 `large`） |
| Processing sample rate | VAD/轉錄流程固定為 `16000` |
| FFmpeg extraction | 單聲道 WAV、`44100 Hz`，並套用 `dynaudnorm=f=100` |
| Lingua detector | 主流程初始化為 `ENGLISH`、`CHINESE`、`JAPANESE`、`ARABIC` |
| Whisper-side filtering helper defaults | 包含 `en`、`zh`、`ja`、`ar`、`yue`、`ko`、`vi`、`es`、`fr` |

假設說明：helper 預設語言列表與主偵測器設定並非完全一致；本 README 依目前實作行為如實保留。

從目前腳本補充的實作細節：

- 執行時會套用 `torch.set_num_threads(1)`。
- VAD 模型透過 `torch.hub.load(...)` 從 `snakers4/silero-vad` 載入。
- 片段清理會移除語言為 `und` 或文字為空的項目。

---

## 📦 Output Format

工具會為每個輸入媒體產生兩種字幕輸出：

- `*.srt`：標準字幕文字，時間戳格式為 `HH:MM:SS,mmm`。
- `*.json`：結構化字幕清單，包含格式化時間戳與語言標記。

典型 JSON 片段結構：

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

說明：

- JSON 輸出中的 `start`/`end` 以 SRT 風格字串序列化。
- `words` 是否存在取決於片段處理/優化階段。
- 對於不確定語言區段，`lang` 可能為 `und`。

---

## 🧪 Examples

對 MP4 執行：

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

對 MOV 執行並強制覆寫：

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

對 FFmpeg 支援的純音訊輸入執行：

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

批次 shell 範例（bash）：

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Development Notes

- 目前正式使用的主腳本是 `vad_lang_subtitle.py`。
- 歷史檔案（`*.old`、`*.shorterlength*`、`archived/`）可供參考，但看起來非正式主線。
- 目前尚未提交封裝專案骨架（`pyproject.toml`、`setup.py`）與 CI/測試套件。
- `data/` 含大型樣本媒體；實驗時請留意儲存庫大小與本機磁碟用量。
- 程式內有 `clean_subtitles_dict()`，但目前主流程未呼叫。
- 目前透過 `--force` 來確保在調整過程中重新產生輸出。

建議本機開發迭代流程：

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

迭代時先用較小模型（`tiny`/`base`/`small`），最終輸出再切換到 `large`。

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | 安裝 FFmpeg，並用 `ffmpeg -version` 驗證。 |
| First run is very slow or appears stuck | 首次下載模型（Whisper + Silero）可能較久；再次執行會更快。 |
| CUDA / GPU errors | 可先以較小 Whisper 模型（`small`、`base`、`tiny`）走 CPU，並確認 PyTorch 版本與環境匹配。 |
| Output files are not regenerated | 使用 `--force` 覆寫既有衍生檔。 |
| `pip install -r requirements.txt` fails because file not found | 請使用 Installation 區塊提供的手動安裝指令。 |
| Inaccurate language tagging on short segments | 在極短或高噪音片段可能發生；目前邏輯已結合 Whisper 與 Lingua，但仍有邊界情況。 |
| Empty or near-empty subtitle output | 確認輸入含語音內容，檢查抽取出的 `.wav`，並在確認 FFmpeg 抽取正常後用 `--force` 重試。 |
| Unexpected language flips between neighboring lines | 極短片段可能出現此現象；可在下游工具依語言與最短時長條件做後處理合併。 |
| FFmpeg extraction fails on damaged media | 腳本會在容器修復（`-c copy -movflags +faststart`）後重試，但嚴重損毀檔案仍可能失敗。 |

快速診斷：

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Known Limitations and Assumptions

- 目前尚未提交相依清單（`requirements.txt`、`pyproject.toml`、`setup.py` 在撰寫當下皆不在儲存庫根目錄）。
- README 宣告授權為 MIT，但目前尚無獨立 `LICENSE` 檔。
- 主流程中的 Lingua 僅明確初始化 `EN/ZH/JA/AR`，而 helper 預設則包含更多候選語言代碼。
- 目前沒有提交自動化測試/基準，因此驗證主要依賴手動流程。
- 根目錄與 `archived/` 含歷史腳本；除非刻意實驗，應只把 `vad_lang_subtitle.py` 視為有效主程式。
- 目前腳本會輸出較詳細的執行日誌與逐片段除錯資訊；這是既有實作的預期行為。

---

## 🗺 Roadmap

- 新增並維護鎖版的 `requirements.txt` 或 `pyproject.toml`。
- 為分段與時間戳清理邏輯加入自動化測試。
- 增加多語邊界案例的效能與品質評估文件。
- 新增可選設定檔支援，取代純程式預設。
- 擴充 `i18n/` README 並維持語言列同步。
- 釐清並統一偵測器設定與 helper 預設間的語言選擇邏輯。
- 新增正式 `LICENSE` 檔，與 README 宣告一致。

---

## 🔗 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) for robust voice activity detection
- [Lingua](https://github.com/pemistahl/lingua-java) for high-accuracy language identification

---

## 🤝 Contributing

1. Fork 並 clone
2. 建立分支：`git checkout -b feat/your-idea`
3. Commit 並 push
4. 開啟 PR

若是較大改動，請附上：

- 預期行為變更的簡短說明
- 可重現的指令範例
- 適用時提供變更前/後字幕片段

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- 請透過 issue 回報 bug、提問使用問題或提出功能需求。
- 贊助與捐款相關請使用上方 support 選項。

---

## 📄 License

MIT © Lachlan Chen
