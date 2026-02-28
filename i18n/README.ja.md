[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

OpenAI Whisper をベースにした、差し替え可能な字幕生成ツールです。複数言語が混在する動画に対して、セグメント単位で高精度な言語検出とリファイン処理を追加した実装です。

> 言語を意識したセグメンテーションにより、実運用の混在言語メディアからよりクリーンな多言語字幕を生成します。

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

| 焦点 | 値 |
| --- | --- |
| 入力 | FFmpeg で扱える音声/動画 |
| パイプライン | VAD 分割 → Whisper 文字起こし → Lingua リファイン |
| 出力 | 正規化済み `*.wav`、`*.srt`、`*.json` |
| 想定用途 | セグメントごとに言語タグを付与した混在言語字幕 |

---

## 目次

- [概要](#-概要)
- [ひと目でわかる概要](#ひと目でわかる概要)
- [主な機能](#-主な機能)
- [パイプラインの流れ](#-パイプラインの流れ)
- [プロジェクト構成](#-プロジェクト構成)
- [前提条件](#-前提条件)
- [インストール](#-インストール)
- [クイックスタート](#-クイックスタート)
- [使い方](#-使い方)
- [設定](#-設定)
- [出力フォーマット](#-出力フォーマット)
- [実行例](#-実行例)
- [開発ノート](#-開発ノート)
- [トラブルシューティング](#-トラブルシューティング)
- [既知の制約と前提](#-既知の制約と前提)
- [ロードマップ](#-ロードマップ)
- [Support](#-support)
- [謝辞](#-謝辞)
- [コントリビュート](#-コントリビュート)
- [Contact](#-contact)
- [ライセンス](#-ライセンス)

---

## ✨ 概要

`MultilingualWhisper` は、[`vad_lang_subtitle.py`](vad_lang_subtitle.py) を中心とした Python CLI パイプラインです。以下を組み合わせています。

- 音声区間分割のための Silero VAD
- 文字起こしと初期言語予測のための OpenAI Whisper
- テキストベースの言語リファインのための Lingua
- 抽出・正規化・メディア処理のための FFmpeg

主な出力は `.srt` と `.json` の字幕ファイル、および抽出・正規化した `*.wav` 音声です。

### ひと目でわかる概要

| 項目 | 詳細 |
|---|---|
| メインエントリーポイント | `vad_lang_subtitle.py` |
| 入力 | FFmpeg が対応する動画/音声 |
| 出力 | `*.wav`, `*.srt`, `*.json` |
| コアフロー | VAD -> Whisper -> Lingua -> リファイン |
| 典型的な用途 | 混在言語向け字幕生成 |

---

## 🚀 主な機能

- **Silero VAD → Whisper パイプライン**
  Voice Activity Detection (VAD) が音声を発話区間に分割し、Whisper が各チャンクを文字起こしします。

- **きめ細かな言語検出**
  [Lingua](https://github.com/pemistahl/lingua-java) を Whisper の言語検出と併用し、各セグメント（必要に応じて単語）に ISO 言語コード (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...) を付与します。

- **インテリジェントなセグメント整形**
  タイムスタンプをクリーンアップしてギャップや重なりを解消します。句読点ベースの分割により、長い文字起こしをカンマ・ピリオド・疑問符などで適切に分割します。VAD で再結合した語を VAD ブロックへ再整列し、より滑らかな字幕にします。長さに応じた分割では言語別の文字数制約も考慮します。

- **多言語字幕出力**
  `.srt` と `.json` の両方を出力し、セグメントごとに言語タグを保持します。これにより、下流のプレイヤーや編集ツールで言語別スタイリングやフィルタが可能です。

- **堅牢なメディア処理**
  FFmpeg で音声を自動抽出・正規化し、壊れたコンテナの修復を試みます。さらに動的ノーマライズ（`dynaudnorm`）で文字起こし品質を改善します。

---

## 🔁 パイプラインの流れ

```text
入力メディア
  -> FFmpeg 抽出 + 正規化 (.wav)
  -> Silero VAD 音声区間タイムスタンプ
  -> Whisper 文字起こし + 言語予測
  -> Lingua によるセグメント言語リファイン
  -> セグメントの統合/分割/タイムスタンプ清掃
  -> 長さに応じた字幕リファイン
  -> .srt と .json を出力
```

`vad_lang_subtitle.py` の主な実行パス:

1. CLI 引数（`--video-path`, `--whisper-model`, `--force`）を解析します。
2. 入力ファイルのベース名から出力パスを決定します。
3. FFmpeg で音声を抽出・正規化します。
4. Silero VAD（`torch.hub`）と Whisper モデルを読み込みます。
5. VAD チャンクで1回目の文字起こしを実行します。
6. セグメントを統合・整形した後、統合区間で2回目の文字起こしを実行します。
7. 字幕長の調整とタイムスタンプのクリーンアップを適用します。
8. `.srt` と `.json` を保存します。

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

> ⚠️ 注記: 以前の README では `requirements.txt` が言及されていましたが、現在はリポジトリのルートに存在しません。

---

## ✅ 前提条件

- Python `3.10+`（近代的な 3.x 環境で検証）
- `ffmpeg` がインストールされ、`PATH` 上にあること
- 選択した Whisper モデルに応じて十分な CPU/GPU と RAM（`large` は GPU 強く推奨）
- 初回実行時に Whisper モデルと Silero VAD アセット（`torch.hub`）を取得するためのインターネット接続

スクリプトで使う主な Python パッケージ:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

動作確認コマンド:

```bash
python --version
ffmpeg -version
```

---

## 🔧 インストール

1. **リポジトリをクローン**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
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

`requirements.txt` がチェックアウト時点で無い場合は、代わりに主要実行依存を手動インストールします。

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

併せてシステムに FFmpeg がインストールされていることを確認してください。

---

## ⚡ クイックスタート

最短でクローンから字幕作成まで進める場合:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

ヒント: 試行段階では `small` を使い、最終品質の生成では `large` に切り替えると効率的です。

入力メディアの隣に生成される成果物:

- `*.wav` 抽出・正規化済み音声
- `*.srt` プレイヤー/編集用字幕ファイル
- `*.json` 構造化多言語字幕メタデータ

---

## 🛠 使い方

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI オプション

| フラグ | 別名 | 必須 | 説明 |
|---|---|---|---|
| `--video-path` | `-t` | Yes | 入力メディアのパス（FFmpeg が対応する動画/音声） |
| `--whisper-model` | — | No | Whisper モデル名（デフォルト: `large`） |
| `--force` | — | No | 既存の `.wav`、`.srt`、`json` があっても再実行 |

### 処理の挙動

- 出力名は入力パスのベース名から決定されます。
- `input.mp4` の場合、`input.wav`（正規化済み音声）、`input.srt`（タイムスタンプ付き字幕）、`input.json`（`start`/`end`/`lang`/`text`、必要に応じて語単位タイミングを含むメタデータ）が生成されます。
- 既存の `.srt` または `.json` がある場合、`--force` を付けない限りスキップされます。

---

## ⚙️ 設定

現在の設定は主に CLI 引数とコード内デフォルトに依存します。

| 設定項目 | 現在の挙動 |
|---|---|
| Whisper model | `--whisper-model`（デフォルト: `large`） |
| 処理サンプルレート | VAD/文字起こし処理のために `16000` 固定 |
| FFmpeg 抽出 | `dynaudnorm=f=100` を使う `44100 Hz` / モノラル WAV |
| Lingua detector | メイン処理では `ENGLISH`、`CHINESE`、`JAPANESE`、`ARABIC` を初期化 |
| Whisper 側フィルタヘルパー既定値 | `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` を含む |

補足: 補助フィルタの既定言語リストとメイン検出器設定は完全に一致しておらず、この README は現行実装の挙動をそのまま反映しています。

---

## 📦 出力フォーマット

入力メディアごとに2種類の字幕成果物を生成します。

- `*.srt`: `HH:MM:SS,mmm` タイムスタンプ形式の標準字幕テキスト
- `*.json`: フォーマット済みタイムスタンプと言語タグを持つ構造化字幕リスト

典型的な JSON セグメント形状:

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

- `start` と `end` は JSON 出力内で SRT 形式文字列として保存されます。
- `words` の有無はセグメント処理/リファイン段階によって異なります。
- 不確実な言語区間では `lang` が `und` になる場合があります。

---

## 🧪 実行例

MP4 に対して実行:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV で実行し、上書きする場合:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

FFmpeg 対応の音声のみ入力に対して実行:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

一括実行例（bash）:

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 開発ノート

- 現行の正規スクリプトは `vad_lang_subtitle.py` です。
- 歴史的ファイル（`*.old`、`*.shorterlength*`、`archived/`）は参照用で、通常は正規版ではありません。
- 現時点ではパッケージング構成（`pyproject.toml`、`setup.py`）や CI/テストスイートはコミットされていません。
- `data/` には大きなサンプルメディアが含まれるため、実験時はリポジトリサイズとローカルディスク使用量に注意してください。
- `clean_subtitles_dict()` はコード内に存在しますが、現行パイプラインでは呼び出されません。
- `--force` は反復的な調整時に出力を確実に再生成する現在の手段です。

推奨ローカル開発フロー:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

反復時は小さいモデル（`tiny`/`base`/`small`）を使い、最終出力では `large` に切り替えて品質を上げることを推奨します。

---

## 🩺 トラブルシューティング

| 症状 | 対処 |
|---|---|
| `ffmpeg: command not found` | FFmpeg をインストールし、`ffmpeg -version` で確認してください。 |
| 初回実行が極端に遅い／停止したように見える | 初回ダウンロード（Whisper + Silero）に時間がかかるため。再実行時は短くなります。 |
| CUDA / GPU エラー | 小さい Whisper モデル（`small`/`base`/`tiny`）で CPU フォールバックを試し、環境に適した PyTorch ビルドを使用してください。 |
| 出力ファイルが更新されない | `--force` を使って既存の派生ファイルを上書きしてください。 |
| `pip install -r requirements.txt` がファイルなしで失敗する | インストール手順の手動コマンドを使用してください。 |
| 短い区間で言語タグが不正確 | 極端に短い/ノイズが多い区間では起こり得ます。Whisper と Lingua を組み合わせた現行ロジックでもエッジケースがあります。 |
| 字幕出力が空、またはほぼ空 | 入力に音声があることを確認し、抽出された `.wav` を確認してから、FFmpeg 抽出検証後に `--force` を付けて再実行してください。 |
| 隣接行で予期せず言語が切り替わる | 非常に短いセグメントで起きることがあります。下流処理で言語ごとおよび最小持続時間での再統合を検討してください。 |

---

## ⚠️ 既知の制約と前提

- 依存ファイル一覧はコミットされていません（`requirements.txt`、`pyproject.toml`、`setup.py` がリポジトリルートに存在しません）。
- ライセンスは README 上で MIT と明記されていますが、現時点で単独の `LICENSE` ファイルはありません。
- Lingua はメインフローで `EN/ZH/JA/AR` のみ初期化されますが、ヘルパー既定値には追加の候補コードが含まれます。
- 自動テストやベンチマークは現在コミットされておらず、検証は主に手動です。
- ルートと `archived/` には履歴スクリプトが残っているため、実験時以外は `vad_lang_subtitle.py` のみをアクティブ版として扱ってください。

---

## 🗺 ロードマップ

- `requirements.txt` または `pyproject.toml` を固定バージョン付きで追加・維持
- セグメント分割とタイムスタンプクリーニングロジック向けの自動テスト追加
- 多言語の境界ケースに対するベンチマークと品質評価ドキュメント追加
- コードデフォルト依存を排した設定ファイルサポートの追加
- `i18n/` の README 多言語セット拡充と言語リンクバーの同期維持
- 検出器設定とヘルパー既定値の言語選択挙動を明確化・統一
- README の宣言と一致する正式な `LICENSE` ファイル追加

---

## 🔗 謝辞

- [OpenAI Whisper](https://github.com/openai/whisper)（音声文字起こし）
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models)（堅牢な音声活動検知）
- [Lingua](https://github.com/pemistahl/lingua-java)（高精度な言語識別）

---

## 🤝 コントリビュート

1. Fork してクローン
2. ブランチ作成: `git checkout -b feat/your-idea`
3. コミットして push
4. PR を作成

大幅な変更の場合:

- 期待する挙動変更の簡潔な説明
- 再現可能な実行コマンド例
- 必要であれば変更前後の字幕スニペット

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- バグ報告、使用方法の質問、機能要望は issue を作成してください。
- スポンサーと寄付については、上記のサポート手段をご利用ください。

## 📄 License

MIT © Lachlan Chen
