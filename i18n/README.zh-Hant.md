[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

一個可直接套用的字幕產生器，基於 OpenAI Whisper，並針對包含混合語言的影片擴充了精準的逐段語言偵測與精修流程。

> 透過語言感知分段，從真實世界的混合語言媒體產生更乾淨的多語字幕。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)

---

## 目錄

- [概覽](#-概覽)
- [快速一覽](#快速一覽)
- [核心功能](#-核心功能)
- [流程管線](#-流程管線)
- [專案結構](#-專案結構)
- [先決條件](#-先決條件)
- [安裝](#-安裝)
- [快速開始](#-快速開始)
- [使用方式](#-使用方式)
- [設定](#-設定)
- [輸出格式](#-輸出格式)
- [範例](#-範例)
- [開發說明](#-開發說明)
- [疑難排解](#-疑難排解)
- [已知限制與前提假設](#-已知限制與前提假設)
- [路線圖](#-路線圖)
- [Support](#-support)
- [致謝](#-致謝)
- [貢獻](#-貢獻)
- [授權](#-授權)

---

## ✨ 概覽

`MultilingualWhisper` 是一個以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 為核心的 Python CLI 管線，整合了：

- Silero VAD 用於語音分段
- OpenAI Whisper 用於轉錄與初步語言預測
- Lingua 用於文字型語言精修
- FFmpeg 用於抽取、正規化與媒體處理

主要輸出為 `.srt` 與 `.json` 字幕檔，以及抽取並正規化後的 `.wav` 音訊。

### 快速一覽

| 項目 | 說明 |
|---|---|
| 主入口 | `vad_lang_subtitle.py` |
| 輸入 | FFmpeg 支援的影片/音訊 |
| 輸出 | `*.wav`, `*.srt`, `*.json` |
| 核心流程 | VAD -> Whisper -> Lingua -> 精修 |
| 典型用途 | 混合語言字幕產生 |

---

## 🚀 核心功能

- **Silero VAD -> Whisper 管線**  
  語音活動偵測（VAD）先將音訊切成語音片段，再由 Whisper 逐段轉錄。

- **細粒度語言偵測**  
  搭配 [Lingua](https://github.com/pemistahl/lingua-java) 與 Whisper 內建偵測器，為每個片段（甚至單字）標註 ISO 語言代碼（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）。

- **智慧片段精修**  
  時間戳清理會避免縫隙與重疊；標點切分會在逗號、句點、問號等位置拆分過長轉錄；VAD 合併會把詞彙重新對齊到 VAD 區塊，使字幕更順暢；同時依語言套用長度感知的分段限制。

- **多語字幕輸出**  
  同時輸出 `.srt` 與 `.json`，並保留逐段語言標籤，方便你在下游播放器或編輯器中依語言套用樣式或篩選。

- **穩健的媒體處理**  
  透過 FFmpeg 自動抽取並正規化音訊、嘗試修復損壞容器，並套用動態正規化（`dynaudnorm`）以提升轉錄清晰度。

---

## 🔁 流程管線

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

1. 解析 CLI 參數（`--video-path`, `--whisper-model`, `--force`）。
2. 依輸入檔名基底推導輸出路徑。
3. 透過 FFmpeg 抽取/正規化音訊。
4. 載入 Silero VAD（`torch.hub`）與 Whisper 模型。
5. 對 VAD 切分後的片段進行第一輪轉錄。
6. 合併/精修片段後，對合併區間進行第二輪轉錄。
7. 套用字幕長度縮減與時間戳清理。
8. 儲存 `.srt` 與 `.json`。

---

## 🗂 專案結構

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

> ⚠️ 注意：先前 README 提到 `requirements.txt`，但目前 repository root 並不存在該檔案。

---

## ✅ 先決條件

- Python `3.10+`（已在現代 3.x 環境測試）
- 已安裝 `ffmpeg`，且可由 `PATH` 存取
- 執行所選 Whisper 模型所需的 CPU/GPU 與 RAM（若使用 `large`，強烈建議 GPU）
- 首次執行需可連網，以下載 Whisper 模型權重與 Silero VAD 資源（`torch.hub`）

此腳本使用的 Python 套件包含：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python 套件）
- `lingua-language-detector`
- `tqdm`

快速驗證指令：

```bash
python --version
ffmpeg -version
```

---

## 🔧 安裝

1. **Clone 此 repo**

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

若你的版本仍缺少 `requirements.txt`，請手動安裝核心執行相依：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

並確認系統層已安裝 FFmpeg。

---

## ⚡ 快速開始

如果你想從 clone 到產生字幕的最快路徑：

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

提示：迭代時先用 `small`，最後再切換到 `large` 以取得最佳品質。

預期會在輸入媒體旁產生：

- `*.wav`：正規化後的抽取音訊
- `*.srt`：供播放器/編輯器使用的字幕檔
- `*.json`：結構化多語字幕中繼資料

---

## 🛠 使用方式

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI 選項

| 旗標 | 別名 | 必填 | 說明 |
|---|---|---|---|
| `--video-path` | `-t` | 是 | 輸入媒體路徑（FFmpeg 支援的影片/音訊） |
| `--whisper-model` | — | 否 | Whisper 模型名稱（預設：`large`） |
| `--force` | — | 否 | 即使 `.wav`、`.srt` 或 `.json` 已存在也強制重跑 |

### 處理行為

- 輸出檔名由輸入基底路徑自動推導。
- 以 `input.mp4` 為例，輸出為 `input.wav`（正規化音訊）、`input.srt`（含時間戳字幕）與 `input.json`（中繼資料，含 `start`、`end`、`lang`、`text`，以及可選詞級時間資訊）。
- 若已有 `.srt` 或 `.json`，預設會跳過；設定 `--force` 才會重跑。

---

## ⚙️ 設定

目前設定主要由 CLI 參數與程式內建預設值共同驅動：

| 設定區塊 | 目前行為 |
|---|---|
| Whisper 模型 | `--whisper-model`（預設 `large`） |
| 處理取樣率 | VAD/轉錄流程硬編碼為 `16000` |
| FFmpeg 抽取 | 單聲道 WAV、`44100 Hz`，並套用 `dynaudnorm=f=100` |
| Lingua 偵測器 | 主流程初始化為 `ENGLISH`、`CHINESE`、`JAPANESE`、`ARABIC` |
| Whisper 側過濾輔助預設 | 包含 `en`、`zh`、`ja`、`ar`、`yue`、`ko`、`vi`、`es`、`fr` |

前提說明：輔助預設中的語言清單與主偵測器設定並非完全一致；本 README 僅忠實描述目前實作行為。

---

## 📦 輸出格式

每個輸入媒體會輸出兩種字幕產物：

- `*.srt`：標準字幕文字，時間戳格式為 `HH:MM:SS,mmm`。
- `*.json`：結構化字幕清單，包含格式化時間戳與語言標籤。

常見 JSON 片段格式：

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

- `start`/`end` 在 JSON 中會序列化為 SRT 風格字串。
- `words` 是否出現取決於片段處理/精修階段。
- 對於語言不確定區段，`lang` 可能為 `und`。

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

批次 shell 範例（bash）：

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 開發說明

- 目前正式主腳本為 `vad_lang_subtitle.py`。
- 歷史檔案（`*.old`、`*.shorterlength*`、`archived/`）可供參考，但看起來非主要版本。
- 目前尚未提交打包專案骨架（`pyproject.toml`、`setup.py`），也沒有 CI/測試套件。
- `data/` 內含大型範例媒體產物；實驗時請留意 repository 體積與本機磁碟使用量。
- 程式中有 `clean_subtitles_dict()`，但目前主流程未呼叫。
- `--force` 是目前用於反覆調參時確保輸出重建的機制。

建議本機開發迴圈：

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

迭代時可先用較小模型（`tiny`/`base`/`small`），最後再切換到 `large` 取得較佳輸出品質。

---

## 🩺 疑難排解

| 症狀 | 建議處理 |
|---|---|
| `ffmpeg: command not found` | 安裝 FFmpeg，並用 `ffmpeg -version` 驗證。 |
| 首次執行很慢或看似卡住 | 首次下載模型（Whisper + Silero）可能較久，後續重跑通常會更快。 |
| CUDA / GPU 錯誤 | 先改用較小 Whisper 模型（`small`、`base`、`tiny`）嘗試 CPU 路徑，並確認 PyTorch 版本與環境相符。 |
| 輸出檔沒有重新產生 | 使用 `--force` 覆寫既有衍生檔。 |
| `pip install -r requirements.txt` 因檔案不存在而失敗 | 改用安裝章節提供的手動相依安裝指令。 |
| 短片段語言標註不準 | 在極短或高噪聲區段常見；目前邏輯雖整合 Whisper 與 Lingua，仍有邊界案例。 |
| 字幕輸出為空或幾乎為空 | 確認輸入確實有語音、檢查抽取出的 `.wav`，並在確認 FFmpeg 抽取正常後加 `--force` 重跑。 |
| 相鄰字幕行語言標籤頻繁跳動 | 在非常短的片段上可能出現；可於下游工具依語言與最短時長條件做後處理合併。 |

---

## ⚠️ 已知限制與前提假設

- 目前尚未提交相依清單（撰寫當下 repository root 缺少 `requirements.txt`、`pyproject.toml`、`setup.py`）。
- README 宣告授權為 MIT，但目前尚無獨立 `LICENSE` 檔案。
- 主流程中 Lingua 明確初始化為 `EN/ZH/JA/AR`，但輔助預設包含更多候選語言代碼。
- 目前未提交自動化測試/基準，因此驗證主要仰賴人工。
- 根目錄與 `archived/` 含有歷史腳本；除非刻意實驗，應以 `vad_lang_subtitle.py` 為主。

---

## 🗺 路線圖

- 新增並維護鎖定版本的 `requirements.txt` 或 `pyproject.toml`。
- 新增分段與時間戳清理邏輯的自動化測試。
- 補上多語邊界案例的效能與品質評估文件。
- 新增可選設定檔支援，取代僅靠程式內預設值。
- 擴充 `i18n/` README 語言集，並保持語言導覽列同步。
- 釐清並統一偵測器設定與輔助預設間的語言選擇行為。
- 新增正式 `LICENSE` 檔，與 README 宣告一致。

---

## ❤️ Support

如果這個專案為你節省了時間，你的支持能幫助專案維護與後續改進。

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

額外支持/社群連結：

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 致謝

- [OpenAI Whisper](https://github.com/openai/whisper) 提供語音轉文字能力
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) 提供穩健的語音活動偵測
- [Lingua](https://github.com/pemistahl/lingua-java) 提供高精度語言辨識

---

## 🤝 貢獻

1. Fork 並 clone
2. 建立分支：`git checkout -b feat/your-idea`
3. Commit 並 push
4. 發起 PR

對於較大幅度修改，請附上：

- 預期行為變更的簡短說明
- 可重現的指令範例
- 在適用時提供修改前/後字幕片段

---

## 📄 授權

MIT © Lachlan Chen
