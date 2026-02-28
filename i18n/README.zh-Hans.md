[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

一个基于 OpenAI Whisper 的即插即用字幕生成器，扩展了逐分段的高精度语言检测与优化，适用于包含混合语言的视频。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents

- [Overview](#-overview)
- [At a Glance](#at-a-glance)
- [Key Features](#-key-features)
- [Pipeline Flow](#-pipeline-flow)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Examples](#-examples)
- [Development Notes](#-development-notes)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations and Assumptions](#-known-limitations-and-assumptions)
- [Roadmap](#-roadmap)
- [Support](#-support)
- [Acknowledgments](#-acknowledgments)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Overview

`MultilingualWhisper` 是一个以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 为核心的 Python CLI 流水线。它结合了：

- 用于语音分段的 Silero VAD
- 用于转写与初步语言预测的 OpenAI Whisper
- 用于基于文本进行语言精修的 Lingua
- 用于提取、归一化和媒体处理的 FFmpeg

主要输出为 `.srt` 与 `.json` 字幕文件，以及提取并归一化后的 `.wav` 音频。

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
  通过语音活动检测（VAD）先将音频切分为语音片段，再由 Whisper 对每个片段进行转写。

- **Fine-grained language detection**
  结合 [Lingua](https://github.com/pemistahl/lingua-java) 与 Whisper 自身语言检测能力，为每个片段（甚至单词级别）标注 ISO 语言代码（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）。

- **Intelligent segment refinement**
  通过时间戳清理避免空隙与重叠；基于标点切分可在逗号、句号、问号等位置拆分长句；VAD 合并会将单词重新对齐回 VAD 片段，以获得更平滑的字幕；并根据语言应用长度感知分段策略。

- **Multilingual subtitles**
  同时输出 `.srt` 与 `.json`，并保留每段语言标签，便于在下游播放器或编辑器中按语言进行样式化或过滤。

- **Robust media handling**
  通过 FFmpeg 自动提取并归一化音频，尝试修复损坏容器，并应用动态归一化（`dynaudnorm`）以提升转写清晰度。

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

`vad_lang_subtitle.py` 中的主要运行路径：

1. 解析 CLI 参数（`--video-path`, `--whisper-model`, `--force`）。
2. 根据输入文件 basename 解析输出路径。
3. 通过 FFmpeg 提取/归一化音频。
4. 加载 Silero VAD（`torch.hub`）和 Whisper 模型。
5. 对 VAD 片段执行第一轮转写。
6. 合并/精修分段后，对合并区间执行第二轮转写。
7. 应用字幕长度优化与时间戳清理。
8. 保存 `.srt` 与 `.json`。

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

> ⚠️ 注意：之前的 README 引用了 `requirements.txt`，但该文件目前在仓库根目录中缺失。

---

## ✅ Prerequisites

- Python `3.10+`（已在现代 3.x 环境测试）
- 已安装 `ffmpeg` 且可在 `PATH` 中访问
- 具备与所选 Whisper 模型匹配的 CPU/GPU 与内存（使用 `large` 时强烈建议 GPU）
- 首次运行需联网下载 Whisper 模型权重与 Silero VAD 资源（`torch.hub`）

脚本使用的 Python 包包括：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python 包）
- `lingua-language-detector`
- `tqdm`

快速验证命令：

```bash
python --version
ffmpeg -version
```

---

## 🔧 Installation

1. **克隆仓库**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **创建并激活虚拟环境**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

如果你的副本中仍缺少 `requirements.txt`，请手动安装核心运行时依赖：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

并确保系统层面已安装 FFmpeg。

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

- 输出文件名由输入基础路径派生。
- 对于 `input.mp4`，输出为 `input.wav`（归一化音频）、`input.srt`（带时间戳字幕）和 `input.json`（包含 `start`、`end`、`lang`、`text` 及可选词级时间信息的元数据）。
- 若已有 `.srt` 或 `.json`，默认会跳过，除非设置了 `--force`。

---

## ⚙️ Configuration

当前配置主要由 CLI 参数和代码默认值驱动：

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

假设说明：辅助默认语言列表与主检测器配置并不完全一致；本 README 按当前实现行为如实保留。

---

## 📦 Output Format

该工具会为每个输入媒体写出两类字幕产物：

- `*.srt`：标准字幕文本，时间戳格式为 `HH:MM:SS,mmm`。
- `*.json`：结构化字幕列表，包含格式化时间戳与语言标签。

典型 JSON 片段结构：

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

说明：

- `start`/`end` 在 JSON 输出中以 SRT 风格字符串序列化。
- `words` 字段是否出现取决于分段处理/优化阶段。
- 对于不确定语言区间，可能出现 `lang` 值为 `und`。

---

## 🧪 Examples

在 MP4 上运行：

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

在 MOV 上运行并强制覆盖：

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

在 FFmpeg 支持的纯音频输入上运行：

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

批处理 shell 示例（bash）：

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Development Notes

- 当前规范且活跃的脚本是 `vad_lang_subtitle.py`。
- 历史文件（`*.old`, `*.shorterlength*`, `archived/`）可供参考，但看起来并非规范入口。
- 当前仓库未提交打包脚手架（`pyproject.toml`, `setup.py`），也没有 CI/测试套件。
- `data/` 包含较大的示例媒体产物，实验时请注意仓库体积与本地磁盘占用。
- 代码中的 `clean_subtitles_dict()` 已存在，但目前未在主流水线中调用。
- `--force` 是当前用于保证输出重新生成、便于迭代调参的机制。

建议的本地开发循环：

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

迭代时建议先使用较小模型（`tiny`/`base`/`small`），最终产出再切换到 `large` 以获得更高质量。

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | 安装 FFmpeg，并用 `ffmpeg -version` 验证。 |
| First run is very slow or appears stuck | 首次模型下载（Whisper + Silero）可能耗时较长；后续重跑会更快。 |
| CUDA / GPU errors | 尝试改用更小 Whisper 模型（`small`, `base`, `tiny`）进行 CPU 回退，并确保 PyTorch 构建与环境匹配。 |
| Output files are not regenerated | 使用 `--force` 覆盖已有派生文件。 |
| `pip install -r requirements.txt` fails because file not found | 使用 Installation 中提供的手动依赖安装命令。 |
| Inaccurate language tagging on short segments | 在极短/高噪片段上可能发生；当前逻辑已结合 Whisper 与 Lingua，但仍有边界场景。 |
| Empty or near-empty subtitle output | 确认输入包含语音，检查提取出的 `.wav`，并在验证 FFmpeg 提取后用 `--force` 重试。 |
| Unexpected language flips between neighboring lines | 在很短片段上可能出现；可在下游工具中按语言与最小时长进行后处理合并。 |

---

## ⚠️ Known Limitations and Assumptions

- 依赖清单尚未提交（`requirements.txt`、`pyproject.toml`、`setup.py` 在编写时的仓库根目录均缺失）。
- README 声明许可证为 MIT，但当前尚无独立 `LICENSE` 文件。
- 主流程中 Lingua 明确初始化为 `EN/ZH/JA/AR`，而辅助默认值包含更多候选代码。
- 当前未提交自动化测试/基准，因此验证主要依赖手动方式。
- 根目录与 `archived/` 中存在历史脚本；除非你有意实验，否则应仅将 `vad_lang_subtitle.py` 视为活跃版本。

---

## 🗺 Roadmap

- 添加并维护固定版本的 `requirements.txt` 或 `pyproject.toml`。
- 为分段与时间戳清理逻辑补充自动化测试。
- 为多语言边界场景补充基准与质量评估文档。
- 增加可选配置文件支持，替代仅靠代码默认值的行为。
- 扩展 `i18n/` 中的 README 多语言集合并保持语言导航同步。
- 统一并澄清检测器配置与辅助默认值之间的语言选择行为。
- 添加正式 `LICENSE` 文件以匹配 README 声明。

---

## 💖 Support

如果这个项目对你有帮助，可以通过以下方式支持开发：

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

对于较大改动，请附带：

- 对预期行为变化的简要说明
- 可复现的命令示例
- 相关时的字幕前后对照片段

---

## 📄 License

MIT © Lachlan Chen
