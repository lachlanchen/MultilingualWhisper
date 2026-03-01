[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

OpenAI Whisper を基盤にしたドロップイン字幕ジェネレーターです。混在言語を含む動画向けに、セグメント単位の高精度な言語検出とリファイン処理を拡張しています。

> 言語対応セグメンテーションにより、実運用の混在言語メディアからよりクリーンな多言語字幕を生成します。

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

> 🌍 **多言語ドキュメント対応**: `i18n/` に英語 + 10 言語の README 翻訳版を用意しています（上部の言語バーから移動可能）。

### ドキュメント言語

| Locale | File |
| --- | --- |

| Focus | Value |
| --- | --- |
| Input | FFmpeg 互換の音声/動画 |
| Pipeline | VAD segmentation -> Whisper transcription -> Lingua refinement |
| Output | 正規化済み `*.wav`、`*.srt`、`*.json` |
| Best use | セグメントごとの言語タグ付き混在言語字幕 |

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

`MultilingualWhisper` は [`vad_lang_subtitle.py`](../vad_lang_subtitle.py) を中心とした Python CLI パイプラインです。以下を組み合わせています。

- Silero VAD による音声区間分割
- OpenAI Whisper による文字起こしと初期言語推定
- Lingua によるテキストベース言語リファイン
- FFmpeg による抽出・正規化・メディア処理

主な出力は `.srt` と `.json` の字幕ファイル、および抽出・正規化された `.wav` 音声です。

### At a Glance

| Item | Details |
|---|---|
| Main entrypoint | `vad_lang_subtitle.py` |
| Input | FFmpeg が対応する動画/音声 |
| Output | `*.wav`, `*.srt`, `*.json` |
| Core flow | VAD -> Whisper -> Lingua -> refinement |
| Typical use case | 混在言語字幕の生成 |

---

## 🚀 Key Features

- **Silero VAD -> Whisper pipeline**  
  Voice Activity Detection (VAD) で音声を発話セグメントに分割し、Whisper が各チャンクを文字起こしします。

- **Fine-grained language detection**  
  [Lingua](https://github.com/pemistahl/lingua-java) と Whisper の言語検出を併用し、各セグメント（単語レベルを含む）に ISO 言語コード（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）を付与します。

- **Intelligent segment refinement**  
  タイムスタンプをクリーンアップしてギャップや重なりを解消します。句読点分割で長い転写をカンマ・ピリオド・疑問符などで分割し、VAD マージで単語を VAD ブロックへ再整列して字幕を滑らかにします。さらに言語ごとの長さ制限を適用します。

- **Multilingual subtitles**  
  `.srt` と `.json` の両方を出力し、セグメントごとの言語タグを保持します。これにより下流のプレイヤーや編集ツールで言語別スタイリング/フィルタリングが可能です。

- **Robust media handling**  
  FFmpeg により音声を自動抽出・正規化し、壊れたコンテナの修復を試行、`dynaudnorm` による動的正規化で転写を明瞭にします。

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

`vad_lang_subtitle.py` における主な実行フロー:

1. CLI 引数（`--video-path`, `--whisper-model`, `--force`）を解析
2. 入力 basename から出力パスを解決
3. FFmpeg で音声を抽出/正規化
4. Silero VAD（`torch.hub`）と Whisper モデルをロード
5. VAD チャンクに対して 1 回目の転写
6. セグメントをマージ/リファインし、マージ後区間に対して 2 回目の転写
7. 字幕長リダクションとタイムスタンプクリーンアップを適用
8. `.srt` と `.json` を保存

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

> ⚠️ 注: 以前の README では `requirements.txt` に言及していますが、現在はリポジトリルートに存在しません。

---

## ✅ Prerequisites

- Python `3.10+`（現行 3.x 環境で検証）
- `ffmpeg` がインストール済みで `PATH` から実行可能
- 選択する Whisper モデルに応じた CPU/GPU と RAM（`large` では GPU を強く推奨）
- 初回実行時に Whisper 重みと Silero VAD アセット（`torch.hub`）を取得するためのインターネット接続

スクリプトが利用する主な Python パッケージ:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

確認用コマンド:

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

チェックアウト時点で `requirements.txt` がない場合は、コア依存関係を手動でインストールしてください:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

あわせて、システムレベルで FFmpeg がインストールされていることを確認してください。

---

## ⚡ Quick Start

クローンから字幕生成までを最短で進める場合:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Tip: 反復作業中は `small` を使い、最終品質が必要な段階で `large` に切り替えてください。

入力メディアと同じ場所に生成される成果物:

- 正規化済み抽出音声 `*.wav`
- プレイヤー/エディタ向け字幕ファイル `*.srt`
- 多言語字幕メタデータを持つ構造化 `*.json`

---

## 🎚 Model Selection Guide

速度と品質の要件に応じて Whisper モデルを選択します:

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | 最速 | 最低 | 高速スモークテストとパイプライン検証 |
| `small` | 高速 | 良好 | 日常的な反復開発とローカル作業 |
| `medium` | 中程度 | より高品質 | バランス重視の本番ワークフロー |
| `large` (default) | 最低速 | 最高 | 最高品質が必要な最終字幕出力 |

実運用での基本パターン:

1. `small --force` で反復
2. タイミングと言語タグを検証
3. 納品用に `large --force` で再実行

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
| `--video-path` | `-t` | Yes | 入力メディアのパス（FFmpeg 対応の動画/音声） |
| `--whisper-model` | — | No | Whisper モデル名（デフォルト: `large`） |
| `--force` | — | No | `.wav`、`.srt`、`.json` が既存でも再実行 |

### Processing Behavior

- 出力ファイル名は入力ベースパスから決定されます。
- `input.mp4` の場合、`input.wav`（正規化音声）、`input.srt`（タイムスタンプ付き字幕）、`input.json`（`start`, `end`, `lang`, `text` と必要に応じて単語タイミングを含むメタデータ）が生成されます。
- 既存の `.srt` または `.json` がある場合、`--force` がないとスキップされます。

---

## ⚙️ Configuration

現在の設定は主に CLI 引数とコード内デフォルトで制御されています:

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | VAD/転写処理向けに `16000` をハードコード |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | メインフローでは `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` で初期化 |
| Whisper-side filtering helper defaults | `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` を含む |

前提に関する注記: ヘルパー既定値の言語リストとメイン検出器の初期化は完全には一致していません。本 README では実装されている現在の挙動をそのまま記載しています。

現行スクリプトの追加実装詳細:

- 実行時に `torch.set_num_threads(1)` が適用されます。
- VAD モデルは `torch.hub.load(...)` を使って `snakers4/silero-vad` から読み込まれます。
- 言語が `und`、またはテキストが空のセグメントはクリーン処理で除去されます。

---

## 📦 Output Format

本ツールは入力メディアごとに 2 つの字幕成果物を出力します:

- `*.srt`: `HH:MM:SS,mmm` タイムスタンプを持つ標準字幕テキスト
- `*.json`: フォーマット済みタイムスタンプと言語タグを持つ構造化字幕リスト

JSON セグメントの典型例:

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

補足:

- JSON 出力では `start`/`end` は SRT 形式の文字列としてシリアライズされます。
- `words` はセグメントの処理/リファイン段階に応じて含まれる場合があります。
- 言語が不確かな区間では `lang` が `und` になることがあります。

---

## 🧪 Examples

MP4 に対して実行:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV に対して強制上書き実行:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

FFmpeg 対応の音声入力に対して実行:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

バッチ処理のシェル例（bash）:

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Development Notes

- 正式なアクティブスクリプトは `vad_lang_subtitle.py` です。
- 履歴ファイル（`*.old`, `*.shorterlength*`, `archived/`）は参照には有用ですが、非カノニカルとみなされます。
- 現在、パッケージング用のひな形（`pyproject.toml`, `setup.py`）および CI/テストスイートはコミットされていません。
- `data/` には大きなサンプルメディア成果物が含まれるため、実験時はリポジトリサイズとローカルディスク容量に注意してください。
- `clean_subtitles_dict()` はコード内に存在しますが、現在メインパイプラインからは呼び出されていません。
- 反復調整時に出力再生成を確実に行う現在の手段は `--force` です。

推奨ローカル開発ループ:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

反復中は小さいモデル（`tiny`/`base`/`small`）を使い、最終出力品質が必要な段階で `large` へ切り替えてください。

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | FFmpeg をインストールし、`ffmpeg -version` で確認してください。 |
| First run is very slow or appears stuck | 初回はモデルダウンロード（Whisper + Silero）に時間がかかることがあります。再実行は速くなります。 |
| CUDA / GPU errors | `small`/`base`/`tiny` など小さい Whisper モデルで CPU フォールバックを試し、環境に合った PyTorch ビルドを確認してください。 |
| Output files are not regenerated | 既存の派生ファイルを上書きするには `--force` を使用してください。 |
| `pip install -r requirements.txt` fails because file not found | Installation セクションに記載の手動インストールコマンドを使用してください。 |
| Inaccurate language tagging on short segments | 非常に短い/ノイズの多い区間で発生することがあります。現在は Whisper と Lingua を併用していますが、依然としてエッジケースがあります。 |
| Empty or near-empty subtitle output | 入力に音声が含まれているか確認し、抽出された `.wav` を点検した上で `--force` で再試行してください。 |
| Unexpected language flips between neighboring lines | 非常に短いセグメントで起こり得ます。下流ツール側で言語と最小長に基づくポストマージを検討してください。 |
| FFmpeg extraction fails on damaged media | スクリプトはコンテナ修復（`-c copy -movflags +faststart`）後に再試行しますが、破損が重いファイルでは失敗する場合があります。 |

簡易診断:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Known Limitations and Assumptions

- 依存関係マニフェストは未コミットです（作成時点で `requirements.txt`, `pyproject.toml`, `setup.py` はリポジトリルートにありません）。
- README では MIT ライセンスと明記されていますが、単独の `LICENSE` ファイルは現時点で存在しません。
- メインフローの Lingua 初期化は `EN/ZH/JA/AR` 明示指定ですが、ヘルパー既定値にはより多くの候補コードが含まれます。
- 自動テスト/ベンチマークは現状コミットされておらず、検証は主に手動です。
- ルートと `archived/` に履歴スクリプトが存在します。意図的に実験する場合を除き、アクティブ扱いは `vad_lang_subtitle.py` のみです。
- 現行実装では詳細なランタイムログとセグメント単位デバッグ出力を表示します。これは想定動作です。

---

## 🗺 Roadmap

- 固定バージョンの `requirements.txt` または `pyproject.toml` を追加・保守
- セグメンテーションおよびタイムスタンプクリーンアップロジックの自動テストを追加
- 多言語エッジケース向けのベンチマーク/品質評価ドキュメントを追加
- コード内デフォルト依存ではなく任意の設定ファイル対応を追加
- `i18n/` の README 言語セットを拡充し、言語バーの同期を維持
- 検出器設定とヘルパー既定値の言語選択挙動を明確化・統一
- README の宣言に合わせた正式な `LICENSE` ファイルを追加

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

大きな変更の場合は、以下を含めてください:

- 期待される挙動変更の短い説明
- 再現可能なコマンド例
- 必要に応じて字幕の before/after スニペット

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- バグ報告、利用方法の質問、機能要望は Issue を作成してください。
- スポンサーや寄付に関する問い合わせは上記サポート手段を利用してください。

---

## 📄 License

MIT © Lachlan Chen
