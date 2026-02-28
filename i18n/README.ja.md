[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

OpenAI Whisper をベースにした、差し替え可能な字幕生成ツールです。複数言語が混在する動画向けに、セグメント単位の高精度な言語検出とリファイン処理を拡張しています。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 目次

- [概要](#-概要)
- [ひと目でわかる概要](#ひと目でわかる概要)
- [主な機能](#-主な機能)
- [パイプラインの流れ](#-パイプラインの流れ)
- [プロジェクト構成](#-プロジェクト構成)
- [前提条件](#-前提条件)
- [インストール](#-インストール)
- [使い方](#-使い方)
- [設定](#-設定)
- [出力フォーマット](#-出力フォーマット)
- [実行例](#-実行例)
- [開発メモ](#-開発メモ)
- [トラブルシューティング](#-トラブルシューティング)
- [既知の制約と前提](#-既知の制約と前提)
- [ロードマップ](#-ロードマップ)
- [サポート](#-サポート)
- [謝辞](#-謝辞)
- [コントリビュート](#-コントリビュート)
- [ライセンス](#-ライセンス)

---

## ✨ 概要

`MultilingualWhisper` は、[`vad_lang_subtitle.py`](vad_lang_subtitle.py) を中心とした Python CLI パイプラインです。以下を組み合わせています。

- 音声区間分割のための Silero VAD
- 文字起こしと初期言語推定のための OpenAI Whisper
- テキストベースの言語リファインのための Lingua
- 抽出・正規化・メディア処理のための FFmpeg

主な出力は `.srt` と `.json` の字幕ファイル、および抽出・正規化した `.wav` 音声です。

### ひと目でわかる概要

| 項目 | 詳細 |
|---|---|
| メインエントリーポイント | `vad_lang_subtitle.py` |
| 入力 | FFmpeg が対応する動画/音声 |
| 出力 | `*.wav`, `*.srt`, `*.json` |
| コアフロー | VAD -> Whisper -> Lingua -> refinement |
| 典型的な用途 | 混在言語向け字幕生成 |

---

## 🚀 主な機能

- **Silero VAD -> Whisper パイプライン**  
  Voice Activity Detection (VAD) で音声を発話セグメントに分割し、各チャンクを Whisper で文字起こしします。

- **きめ細かな言語検出**  
  Whisper の言語検出に加えて [Lingua](https://github.com/pemistahl/lingua-java) を使用し、各セグメント（単語単位を含む）に ISO 言語コード（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）を付与します。

- **インテリジェントなセグメント整形**  
  タイムスタンプを補正してギャップや重なりを解消します。句読点ベースの分割で長い文字起こしをカンマ・ピリオド・疑問符などで区切り、VAD マージで単語を VAD ブロックへ再整列して、より読みやすい字幕にします。さらに言語別の文字数制限を考慮した分割も適用します。

- **多言語字幕出力**  
  `.srt` と `.json` の両方を出力し、セグメントごとの言語タグを保持します。これにより、後段のプレイヤーやエディタで言語別のスタイル適用やフィルタリングが可能です。

- **堅牢なメディア処理**  
  FFmpeg で音声を自動抽出・正規化し、壊れたコンテナの修復も試みます。さらに動的正規化（`dynaudnorm`）で文字起こし品質を高めます。

---

## 🔁 パイプラインの流れ

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

`vad_lang_subtitle.py` の主な実行パス:

1. CLI 引数（`--video-path`, `--whisper-model`, `--force`）を解析。
2. 入力ファイルの basename から出力パスを解決。
3. FFmpeg で音声を抽出・正規化。
4. Silero VAD（`torch.hub`）と Whisper モデルを読み込み。
5. VAD チャンクに対して 1 回目の文字起こしを実行。
6. セグメントを統合・整形し、統合後区間で 2 回目の文字起こしを実行。
7. 字幕長の調整とタイムスタンプのクリーニングを適用。
8. `.srt` と `.json` を保存。

---

## 🗂 プロジェクト構成

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

> ⚠️ 注記: 以前の README では `requirements.txt` に言及していましたが、現在はリポジトリのルートに存在しません。

---

## ✅ 前提条件

- Python `3.10+`（最新の 3.x 環境で検証）
- `ffmpeg` がインストール済みで、`PATH` から実行可能
- 選択する Whisper モデルに応じた十分な CPU/GPU と RAM（`large` では GPU を強く推奨）
- 初回実行時に、Whisper モデル重みと Silero VAD アセット（`torch.hub`）を取得するためのインターネット接続

スクリプトで使用する主な Python パッケージ:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

簡易確認コマンド:

```bash
python --version
ffmpeg -version
```

---

## 🔧 インストール

1. **このリポジトリをクローン**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **仮想環境を作成して有効化**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **依存関係をインストール**

```bash
pip install -r requirements.txt
```

チェックアウトした環境に `requirements.txt` がない場合は、コア実行依存を手動でインストールしてください。

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

加えて、システムレベルで FFmpeg がインストールされていることを確認してください。

---

## 🛠 使い方

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI オプション

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### 処理の挙動

- 出力ファイル名は入力パスのベース名から決まります。
- `input.mp4` の場合、`input.wav`（正規化音声）、`input.srt`（タイムスタンプ付き字幕）、`input.json`（`start`, `end`, `lang`, `text`、必要に応じて単語タイミングを含むメタデータ）を生成します。
- `--force` を付けない限り、既存の `.srt` または `.json` がある場合はスキップされます。

---

## ⚙️ 設定

現在の設定は主に CLI 引数とコード内デフォルトに依存しています。

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

前提メモ: ヘルパー側デフォルトの言語リストとメイン検出器の設定は完全には一致していません。この README は実装されている現在の挙動をそのまま記載しています。

---

## 📦 出力フォーマット

このツールは、入力メディアごとに 2 種類の字幕成果物を生成します。

- `*.srt`: `HH:MM:SS,mmm` タイムスタンプ付きの標準字幕テキスト
- `*.json`: フォーマット済みタイムスタンプと言語タグを含む構造化字幕リスト

典型的な JSON セグメント:

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

- JSON 出力の `start`/`end` は SRT 形式の文字列としてシリアライズされます。
- `words` はセグメントの処理・整形段階によって含まれる場合があります。
- 言語が不確かな区間では `lang` が `und` になることがあります。

---

## 🧪 実行例

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

バッチ実行例（bash）:

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 開発メモ

- 現在の正規のアクティブスクリプトは `vad_lang_subtitle.py` です。
- 履歴ファイル（`*.old`, `*.shorterlength*`, `archived/`）は参照用として有用ですが、正規系ではないようです。
- 現時点ではパッケージ化向けの設定（`pyproject.toml`, `setup.py`）と CI/テストスイートはコミットされていません。
- `data/` には大きいサンプルメディア成果物が含まれるため、実験時はリポジトリサイズとローカルディスク使用量に注意してください。
- `clean_subtitles_dict()` はコード内に存在しますが、現在メインパイプラインからは呼び出されていません。
- `--force` は、反復チューニング時に出力再生成を確実に行うための現行手段です。

推奨ローカル開発ループ:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

反復中は軽量モデル（`tiny`/`base`/`small`）を使い、最終品質を出す段階で `large` に切り替える運用が有効です。

---

## 🩺 トラブルシューティング

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

## ⚠️ 既知の制約と前提

- 依存マニフェストがコミットされていません（執筆時点でリポジトリルートに `requirements.txt`, `pyproject.toml`, `setup.py` はありません）。
- README ではライセンスを MIT と記載していますが、現時点で独立した `LICENSE` ファイルは存在しません。
- メインフローでは Lingua が `EN/ZH/JA/AR` で明示初期化される一方、ヘルパー側デフォルトにはさらに多くの候補コードが含まれます。
- 自動テスト/ベンチマークはまだコミットされていないため、検証は主に手動です。
- ルートおよび `archived/` に履歴スクリプトが存在します。意図的な実験でない限り、アクティブ対象は `vad_lang_subtitle.py` のみとして扱ってください。

---

## 🗺 ロードマップ

- バージョン固定された `requirements.txt` または `pyproject.toml` を追加・維持する。
- セグメンテーションとタイムスタンプ整形ロジックの自動テストを追加する。
- 多言語の難ケース向けに、ベンチマークと品質評価ドキュメントを整備する。
- コードデフォルト依存ではなく、オプション設定ファイルを追加する。
- `i18n/` の README 言語セットを拡充し、言語バーを同期して維持する。
- 検出器設定とヘルパーデフォルトの言語選択挙動を明確化・統一する。
- README の記載と一致する正式な `LICENSE` ファイルを追加する。

---

## 💖 サポート

このプロジェクトが役立った場合、以下から開発を支援できます。

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 謝辞

- 音声認識: [OpenAI Whisper](https://github.com/openai/whisper)
- 高精度な音声区間検出: [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models)
- 高精度な言語識別: [Lingua](https://github.com/pemistahl/lingua-java)

---

## 🤝 コントリビュート

1. Fork して clone
2. ブランチを作成: `git checkout -b feat/your-idea`
3. コミットして push
4. PR を作成

大きな変更を行う場合は、以下を含めてください。

- 期待される挙動変更の短い説明
- 再現可能なコマンド例
- 必要に応じて字幕の before/after スニペット

---

## 📄 ライセンス

MIT © Lachlan Chen
