[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

一個以 OpenAI Whisper 為基礎、可即插即用的字幕產生器，並針對混合語言影片擴充了精準的逐段語言偵測與優化能力。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ 概覽

`MultilingualWhisper` 是一條以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 為核心的 Python CLI 流程，整合了：

- Silero VAD 用於語音分段
- OpenAI Whisper 用於轉錄與初始語言預測
- Lingua 用於文字層級的語言精修
- FFmpeg 用於擷取、正規化與媒體處理

主要輸出為 `.srt` 與 `.json` 字幕檔，以及擷取並正規化後的 `.wav` 音訊。

### 快速總覽

| 項目 | 說明 |
|---|---|
| 主入口 | `vad_lang_subtitle.py` |
| 輸入 | FFmpeg 支援的影片/音訊 |
| 輸出 | `*.wav`, `*.srt`, `*.json` |
| 核心流程 | VAD -> Whisper -> Lingua -> refinement |
| 常見用途 | 混合語言字幕產生 |

---

## 🚀 核心功能

- **Silero VAD -> Whisper 流程**  
  先透過語音活動偵測（VAD）切分音訊，再由 Whisper 轉錄每個區塊。

- **細粒度語言偵測**  
  結合 [Lingua](https://github.com/pemistahl/lingua-java) 與 Whisper 內建偵測器，為每個片段（甚至單字）標註 ISO 語言代碼（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）。

- **智慧分段精修**  
  時間戳清理可避免間隔或重疊；標點切分會在逗號、句號、問號等位置拆開過長轉錄；VAD 重新合併會把單字對齊回 VAD 區塊，讓字幕更順暢；同時也會依語言套用長度限制。

- **多語字幕輸出**  
  同時輸出 `.srt` 與 `.json`，保留每段語言標籤，方便你在後續播放器或編輯器中依語言篩選或套用樣式。

- **穩健的媒體處理**  
  透過 FFmpeg 自動擷取並正規化音訊、嘗試修復損壞容器，並套用動態正規化（`dynaudnorm`）以提升轉錄清晰度。

---

## 🗂 專案結構

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

> ⚠️ 注意：先前 README 提到 `requirements.txt`，但目前在儲存庫根目錄中缺少此檔案。

---

## ✅ 先決條件

- Python `3.10+`（已在現代 3.x 環境測試）
- 已安裝 `ffmpeg`，且可於 `PATH` 中存取
- 依所選 Whisper 模型準備足夠的 CPU/GPU 與記憶體（若使用 `large`，強烈建議 GPU）
- 首次執行需可連網，以下載 Whisper 權重與 Silero VAD 資源（`torch.hub`）

腳本使用的 Python 套件包含：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python 套件）
- `lingua-language-detector`
- `tqdm`

---

## 🔧 安裝

1. **複製此儲存庫**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **建立並啟用虛擬環境**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **安裝相依套件**

```bash
pip install -r requirements.txt
```

如果你的版本仍缺少 `requirements.txt`，請手動安裝核心執行相依套件：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

並確認系統層級已安裝 FFmpeg。

---

## 🛠 使用方式

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI 選項

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### 處理行為

- 輸出檔名會由輸入路徑的基底名稱推導。
- 對於 `input.mp4`，輸出為 `input.wav`（正規化音訊）、`input.srt`（含時間戳字幕）與 `input.json`（包含 `start`、`end`、`lang`、`text`，以及可選單字時間資訊的中繼資料）。
- 若 `.srt` 或 `.json` 已存在，預設會跳過，除非設定 `--force`。

---

## ⚙️ 設定

目前設定主要由 CLI 與程式內預設值驅動：

- Whisper 模型：`--whisper-model`（預設 `large`）
- 取樣率：處理流程硬編碼為 `16000`
- FFmpeg 擷取：單聲道 WAV、`44100 Hz`，並使用 `dynaudnorm=f=100`
- Lingua 偵測器：主流程初始化為 `ENGLISH`、`CHINESE`、`JAPANESE`、`ARABIC`
- Whisper 端過濾允許的語言代碼，在輔助預設中包含 `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`

假設說明：輔助預設中的語言清單與主偵測器設定並不完全一致；本 README 依目前實作行為如實保留。

---

## 🧪 範例

在 MP4 上執行：

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

在 MOV 上執行並強制覆寫：

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

在 FFmpeg 支援的純音訊輸入上執行：

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 開發備註

- 目前的正式主腳本是 `vad_lang_subtitle.py`。
- 歷史檔案（`*.old`、`*.shorterlength*`、`archived/`）可供參考，但看起來並非正式版本。
- 目前尚未提交套件化專案骨架（`pyproject.toml`、`setup.py`），也沒有 CI/測試套件。
- `data/` 含有大型範例媒體產物；實驗時請留意儲存庫體積與本機磁碟用量。
- `clean_subtitles_dict()` 存在於程式中，但目前未由主流程呼叫。

---

## 🩺 疑難排解

| 症狀 | 建議作法 |
|---|---|
| `ffmpeg: command not found` | Install FFmpeg and verify with `ffmpeg -version`. |
| First run is very slow or appears stuck | Initial model downloads (Whisper + Silero) can take time; reruns are faster. |
| CUDA / GPU errors | Try CPU fallback by using a smaller Whisper model (`small`, `base`, `tiny`) and ensure matching PyTorch build for your environment. |
| Output files are not regenerated | Use `--force` to overwrite existing derived files. |
| `pip install -r requirements.txt` fails because file not found | Use manual dependency install command shown in Installation. |
| Inaccurate language tagging on short segments | This can happen on extremely short/noisy spans; current logic combines Whisper and Lingua but still has edge cases. |

---

## 🗺 路線圖

- 新增並維護固定版本的 `requirements.txt` 或 `pyproject.toml`。
- 為分段與時間戳清理邏輯新增自動化測試。
- 新增多語邊界情境的基準與品質評估文件。
- 新增可選設定檔支援，而非僅依賴程式內預設值。
- 擴充 `i18n/` 內 README 語言集合，並保持語言導覽列同步。

---

## 💖 支援

如果這個專案對你有幫助，可以透過以下方式支持開發：

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 致謝

- [OpenAI Whisper](https://github.com/openai/whisper)（語音轉文字）
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models)（穩健的語音活動偵測）
- [Lingua](https://github.com/pemistahl/lingua-java)（高準確率語言識別）

---

## 🤝 貢獻

1. Fork 並 clone
2. 建立分支：`git checkout -b feat/your-idea`
3. Commit 並 push
4. 開啟 PR

---

## 📄 授權

MIT © Lachlan Chen
