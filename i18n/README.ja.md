[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

OpenAI Whisper を基盤としたドロップイン型の字幕生成ツールです。複数言語が混在する動画向けに、セグメント単位で高精度な言語検出とリファイン処理を拡張しています。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ 概要

`MultilingualWhisper` は [`vad_lang_subtitle.py`](vad_lang_subtitle.py) を中心とした Python CLI パイプラインです。次を組み合わせています。

- 音声区間分割のための Silero VAD
- 文字起こしと初期言語推定のための OpenAI Whisper
- テキストベース言語リファインのための Lingua
- 抽出・正規化・メディア処理のための FFmpeg

主な出力は `.srt` と `.json` の字幕ファイル、および抽出・正規化された `.wav` 音声です。

### ひと目でわかる情報

| 項目 | 詳細 |
|---|---|
| メインエントリーポイント | `vad_lang_subtitle.py` |
| 入力 | FFmpeg が対応する動画/音声 |
| 出力 | `*.wav`, `*.srt`, `*.json` |
| コアフロー | VAD -> Whisper -> Lingua -> refinement |
| 典型的な用途 | 混在言語の字幕生成 |

---

## 🚀 主な機能

- **Silero VAD -> Whisper パイプライン**  
  Voice Activity Detection (VAD) で音声を発話セグメントに分割し、その後 Whisper が各チャンクを文字起こしします。

- **高粒度な言語検出**  
  Whisper 自身の言語検出に加えて [Lingua](https://github.com/pemistahl/lingua-java) を使用し、各セグメント（単語単位を含む）に ISO 言語コード（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）を付与します。

- **インテリジェントなセグメントリファイン**  
  タイムスタンプをクリーンアップして欠損や重なりを防ぎます。カンマ・ピリオド・疑問符などの句読点で長い文字起こしを分割します。VAD マージにより語を VAD ブロックへ再整列し、字幕の自然さを向上させます。さらに言語ごとの長さ制限を考慮したセグメンテーションを行います。

- **多言語字幕出力**  
  `.srt` と `.json` の両方を出力し、セグメントごとの言語タグを保持します。これにより、後段のプレイヤーやエディタで言語別のスタイル適用やフィルタが可能です。

- **堅牢なメディア処理**  
  FFmpeg により音声を自動抽出・正規化し、破損コンテナの修復を試行します。さらに動的正規化（`dynaudnorm`）を適用して、より明瞭な文字起こしを目指します。

---

## 🗂 プロジェクト構成

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

> ⚠️ 注: 以前の README では `requirements.txt` に言及していましたが、現在はリポジトリルートに存在しません。

---

## ✅ 前提条件

- Python `3.10+`（最新の 3.x 環境で検証）
- `ffmpeg` がインストール済みで `PATH` から利用可能
- 選択した Whisper モデルを実行するのに十分な CPU/GPU と RAM（`large` では GPU を強く推奨）
- 初回実行時に Whisper モデル重みと Silero VAD アセット（`torch.hub`）を取得するためのインターネット接続

スクリプトで使用する Python パッケージは以下です。

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

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

チェックアウト環境に `requirements.txt` がまだない場合は、コア実行時依存を手動でインストールしてください。

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

あわせて、システムレベルで FFmpeg がインストールされていることを確認してください。

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

- 出力ファイル名は入力パスのベース名から生成されます。
- `input.mp4` の場合、出力は `input.wav`（正規化音声）、`input.srt`（タイムスタンプ付き字幕）、`input.json`（`start`, `end`, `lang`, `text`、必要に応じて単語タイミングを含むメタデータ）です。
- 既存の `.srt` または `.json` がある場合、`--force` 未指定ではスキップされます。

---

## ⚙️ 設定

現在の設定は主に CLI 引数とコード内デフォルトによって決まります。

- Whisper model: `--whisper-model` (default `large`)
- Sampling rate: hard-coded to `16000` for processing
- FFmpeg extraction: mono WAV, `44100 Hz`, with `dynaudnorm=f=100`
- Lingua detector: initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow
- Allowed language codes for Whisper-side filtering include `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` in helper defaults

前提に関する注記: ヘルパー側デフォルトの言語リストとメイン検出器の設定は完全には一致していません。この README は、実装されている現行挙動をそのまま記載しています。

---

## 🧪 例

MP4 に対して実行:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV に対して実行し、上書きを強制:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

FFmpeg 対応の音声入力に対して実行:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 開発ノート

- 現在の正準スクリプトは `vad_lang_subtitle.py` です。
- 履歴ファイル（`*.old`, `*.shorterlength*`, `archived/`）は参照には有用ですが、正準ではないようです。
- 現時点ではパッケージング用の構成（`pyproject.toml`, `setup.py`）や CI/テストスイートはコミットされていません。
- `data/` には大きなサンプルメディア成果物が含まれるため、実験時はリポジトリサイズとローカルディスク使用量に注意してください。
- `clean_subtitles_dict()` はコード内に存在しますが、現在メインパイプラインでは呼び出されていません。

---

## 🩺 トラブルシューティング

| 症状 | 対処 |
|---|---|
| `ffmpeg: command not found` | FFmpeg をインストールし、`ffmpeg -version` で確認してください。 |
| 初回実行が非常に遅い、または停止しているように見える | 初回のモデルダウンロード（Whisper + Silero）には時間がかかることがあります。再実行は高速になります。 |
| CUDA / GPU エラー | 小さい Whisper モデル（`small`, `base`, `tiny`）で CPU フォールバックを試し、環境に合った PyTorch ビルドを使用してください。 |
| 出力ファイルが再生成されない | 既存の派生ファイルを上書きするには `--force` を使用してください。 |
| `pip install -r requirements.txt` が file not found で失敗する | インストール節にある手動依存インストールコマンドを使用してください。 |
| 短いセグメントで言語タグが不正確 | 極端に短い/ノイズの多い区間では発生し得ます。現在のロジックは Whisper と Lingua を組み合わせていますが、依然としてエッジケースがあります。 |

---

## 🗺 ロードマップ

- 固定バージョン管理された `requirements.txt` または `pyproject.toml` を追加・維持する。
- セグメンテーションとタイムスタンプクリーンアップロジックの自動テストを追加する。
- 多言語のエッジケース向けに、ベンチマークと品質評価ドキュメントを追加する。
- コード内デフォルト依存ではなく、任意設定ファイルをサポートする。
- `i18n/` の README セットを拡充し、言語バーの同期を維持する。

---

## 💖 サポート

このプロジェクトが役に立った場合、以下から開発を支援できます。

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 謝辞

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) for robust voice activity detection
- [Lingua](https://github.com/pemistahl/lingua-java) for high-accuracy language identification

---

## 🤝 コントリビューション

1. Fork and clone
2. Create a branch: `git checkout -b feat/your-idea`
3. Commit and push
4. Open a PR

---

## 📄 ライセンス

MIT © Lachlan Chen
