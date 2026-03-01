[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

一个基于 OpenAI Whisper 的即插即用字幕生成器，扩展了精确到分段级别的语言检测与优化，适用于包含混合语言的视频。

> 通过具备语言感知能力的分段，从真实世界的混合语言媒体中生成更干净的多语言字幕。

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


### 文档语言

| Locale | File |
| --- | --- |

| Focus | Value |
| --- | --- |
| Input | FFmpeg-compatible audio/video |
| Pipeline | VAD segmentation -> Whisper transcription -> Lingua refinement |
| Output | Normalized `*.wav`, `*.srt`, and `*.json` |
| Best use | Mixed-language subtitles with per-segment language tags |

---

## Table of Contents

- [概览](#-概览)
- [快速浏览](#快速浏览)
- [核心特性](#-核心特性)
- [流水线流程](#-流水线流程)
- [项目结构](#-项目结构)
- [先决条件](#-先决条件)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [模型选择指南](#-模型选择指南)
- [使用方法](#-使用方法)
- [配置](#-配置)
- [输出格式](#-输出格式)
- [示例](#-示例)
- [开发说明](#-开发说明)
- [故障排查](#-故障排查)
- [已知限制与假设](#-已知限制与假设)
- [路线图](#-路线图)
- [致谢](#-致谢)
- [贡献](#-贡献)
- [支持](#-support)
- [联系](#-联系)
- [许可证](#-许可证)

---

## ✨ 概览

`MultilingualWhisper` 是一个以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 为核心的 Python CLI 流水线。它结合了：

- 用于语音分段的 Silero VAD
- 用于转写与初始语言预测的 OpenAI Whisper
- 用于文本语言细化的 Lingua
- 用于提取、归一化和媒体处理的 FFmpeg

主要输出包括 `.srt` 和 `.json` 字幕文件，以及提取并归一化的 `.wav` 音频。

### 快速浏览

| Item | Details |
|---|---|
| Main entrypoint | `vad_lang_subtitle.py` |
| Input | Video/audio supported by FFmpeg |
| Output | `*.wav`, `*.srt`, `*.json` |
| Core flow | VAD -> Whisper -> Lingua -> refinement |
| Typical use case | Mixed-language subtitle generation |

---

## 🚀 核心特性

- **Silero VAD -> Whisper 流水线**  
  语音活动检测（VAD）先将音频切分为语音片段，然后 Whisper 对每个片段进行转写。

- **细粒度语言检测**  
  使用 [Lingua](https://github.com/pemistahl/lingua-java) 并结合 Whisper 自带检测器，为每个片段（甚至单词）打上 ISO 语言代码（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）。

- **智能分段优化**  
  时间戳清理可确保没有空隙或重叠。标点切分会在逗号、句号、问号等位置切开较长转写。VAD 合并会把词重新对齐到 VAD 块，以获得更平滑的字幕。长度感知分段会应用语言相关限制。

- **多语言字幕**  
  同时输出 `.srt` 和 `.json`，并保留每段的语言标签，便于你在下游播放器或编辑器中按语言样式化或过滤。

- **稳健的媒体处理**  
  通过 FFmpeg 自动提取并归一化音频，尝试修复损坏容器，并应用动态归一化（`dynaudnorm`）以获得更清晰的转写。

---

## 🔁 流水线流程

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

`vad_lang_subtitle.py` 中的主运行路径：

1. 解析 CLI 参数（`--video-path`, `--whisper-model`, `--force`）。
2. 根据输入文件基础名解析输出路径。
3. 通过 FFmpeg 提取并归一化音频。
4. 加载 Silero VAD（`torch.hub`）和 Whisper 模型。
5. 在 VAD 分块上执行首轮转写。
6. 合并/细化分段，然后在合并跨度上执行第二轮转写。
7. 应用字幕长度压缩和时间戳清理。
8. 保存 `.srt` 和 `.json`。

---

## 🗂 项目结构

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

> ⚠️ 注意：先前 README 提到了 `requirements.txt`，但它目前缺失于仓库根目录。

---

## ✅ 先决条件

- Python `3.10+`（已在现代 3.x 环境中测试）
- 已安装 `ffmpeg` 且可在 `PATH` 中访问
- 与所选 Whisper 模型匹配的充足 CPU/GPU + RAM（`large` 强烈建议使用 GPU）
- 首次运行需要联网以拉取 Whisper 模型权重和 Silero VAD 资源（`torch.hub`）

脚本使用的 Python 包包括：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python package）
- `lingua-language-detector`
- `tqdm`

快速验证命令：

```bash
python --version
ffmpeg -version
```

---

## 🔧 安装

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

如果你的 checkout 中仍然缺少 `requirements.txt`，请手动安装核心运行依赖：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

并确保系统层面已安装 FFmpeg。

---

## ⚡ 快速开始

如果你想从 clone 到字幕生成走最快路径：

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

提示：迭代时先用 `small`，最终质量导出再切换到 `large`。

在输入媒体旁边你会看到这些产物：

- `*.wav` 归一化后的提取音频
- `*.srt` 供播放器/编辑器使用的字幕文件
- `*.json` 结构化多语言字幕元数据

---

## 🎚 模型选择指南

根据速度与质量目标选择 Whisper 模型：

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | Fastest | Lowest | Fast smoke tests and pipeline validation |
| `small` | Fast | Good | Daily iteration and local development |
| `medium` | Medium | Better | Balanced production workflows |
| `large` (default) | Slowest | Best | Final subtitle exports for highest quality |

实用模式：

1. 用 `small --force` 进行迭代
2. 验证时间轴和语言标签
3. 用 `large --force` 重新运行以产出交付版本

---

## 🛠 使用方法

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI 选项

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | 输入媒体路径（FFmpeg 支持的视频/音频） |
| `--whisper-model` | — | No | Whisper 模型名称（默认：`large`） |
| `--force` | — | No | 即使 `.wav`、`.srt` 或 `.json` 已存在也重新运行 |

### 处理行为

- 输出名称由输入基础路径派生。
- 对于 `input.mp4`，输出为 `input.wav`（归一化音频）、`input.srt`（带时间戳字幕）和 `input.json`（包含 `start`, `end`, `lang`, `text`，以及可选词级时间信息的元数据）。
- 若存在 `.srt` 或 `.json`，则会跳过，除非设置 `--force`。

---

## ⚙️ 配置

当前配置主要由 CLI 参数和代码默认值驱动：

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

假设说明：辅助函数默认语言列表和主检测器初始化配置并不完全一致；本 README 按当前实现保留该行为。

来自当前脚本的额外实现细节：

- 运行时会应用 `torch.set_num_threads(1)`。
- VAD 模型通过 `torch.hub.load(...)` 从 `snakers4/silero-vad` 加载。
- 分段清理会移除语言为 `und` 或文本为空的条目。

---

## 📦 输出格式

该工具会为每个输入媒体写出两个字幕产物：

- `*.srt`：标准字幕文本，时间戳格式为 `HH:MM:SS,mmm`。
- `*.json`：结构化字幕列表，包含格式化时间戳和语言标签。

典型 JSON 分段结构：

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

- `start`/`end` 在 JSON 输出中会序列化为 SRT 风格字符串。
- `words` 是否出现取决于分段处理/细化阶段。
- 对于不确定语言跨度，可能会出现 `lang` 值为 `und`。

---

## 🧪 示例

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

## 🧭 开发说明

- 当前规范使用的脚本是 `vad_lang_subtitle.py`。
- 历史文件（`*.old`, `*.shorterlength*`, `archived/`）可用于参考，但看起来不是规范主线。
- 目前尚未提交打包工程脚手架（`pyproject.toml`, `setup.py`），也没有 CI/测试套件。
- `data/` 包含较大的示例媒体产物；实验时请注意仓库体积和本地磁盘占用。
- `clean_subtitles_dict()` 存在于代码中，但当前主流水线未调用。
- `--force` 是当前用于确保重新生成输出、支持迭代调参的机制。

建议的本地开发循环：

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

迭代阶段使用更小模型（`tiny`/`base`/`small`），最终质量导出再切换到 `large`。

---

## 🩺 故障排查

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
| FFmpeg extraction fails on damaged media | Script retries after container repair (`-c copy -movflags +faststart`), but heavily corrupted files may still fail. |

快速诊断：

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ 已知限制与假设

- 当前仓库根目录未提交依赖清单（`requirements.txt`, `pyproject.toml`, `setup.py` 缺失）。
- README 中声明许可证为 MIT，但当前并不存在独立的 `LICENSE` 文件。
- 主流程中 Lingua 显式初始化为 `EN/ZH/JA/AR`，而辅助函数默认值包含更多候选代码。
- 目前未提交自动化测试/基准，因此验证主要依赖手动方式。
- 根目录和 `archived/` 中存在历史脚本；除非有意实验，否则应仅将 `vad_lang_subtitle.py` 视为活跃实现。
- 脚本当前会输出详细运行日志和逐段调试信息；这是当前实现下的预期行为。

---

## 🗺 路线图

- 添加并维护固定版本的 `requirements.txt` 或 `pyproject.toml`。
- 为分段与时间戳清理逻辑添加自动化测试。
- 为多语言边缘场景补充基准和质量评估文档。
- 增加可选配置文件支持，而非仅依赖代码默认值。
- 扩展 `i18n/` 中的 README 语言集合，并保持语言栏同步。
- 澄清并统一检测器配置与辅助默认值之间的语言选择行为。
- 添加正式的 `LICENSE` 文件以匹配 README 声明。

---

## 🔗 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) for robust voice activity detection
- [Lingua](https://github.com/pemistahl/lingua-java) for high-accuracy language identification

---

## 🤝 贡献

1. Fork 并 clone
2. 创建分支：`git checkout -b feat/your-idea`
3. 提交并推送
4. 发起 PR

对于较大的改动，请附带：

- 预期行为变化的简短说明
- 可复现的命令示例
- 相关时提供字幕前后对比片段

---

## 📫 联系

- 如需反馈 bug、使用问题或功能请求，请提交 issue。
- 赞助或捐赠相关咨询请使用上方支持选项。

---

## 📄 许可证

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
