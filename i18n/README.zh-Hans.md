[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

一个基于 OpenAI Whisper 的即插即用字幕生成器，并扩展了精细到分段级别的语言检测与优化，适用于包含混合语言的视频。

> 通过语言感知分段，从真实世界的混合语言媒体中生成更干净的多语言字幕。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)

---

## 目录

- [概览](#-概览)
- [快速了解](#快速了解)
- [核心特性](#-核心特性)
- [流水线流程](#-流水线流程)
- [项目结构](#-项目结构)
- [前置要求](#-前置要求)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [用法](#-用法)
- [配置](#-配置)
- [输出格式](#-输出格式)
- [示例](#-示例)
- [开发说明](#-开发说明)
- [故障排查](#-故障排查)
- [已知限制与假设](#-已知限制与假设)
- [路线图](#-路线图)
- [支持](#-support)
- [致谢](#-致谢)
- [贡献](#-贡献)
- [许可](#-许可)

---

## ✨ 概览

`MultilingualWhisper` 是一个以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 为核心的 Python CLI 流水线。它组合了：

- 使用 Silero VAD 做语音分段
- 使用 OpenAI Whisper 做转写与初始语言预测
- 使用 Lingua 做基于文本的语言细化
- 使用 FFmpeg 做提取、归一化和媒体处理

主要输出为 `.srt` 与 `.json` 字幕文件，以及提取并归一化后的 `.wav` 音频。

### 快速了解

| 项目 | 说明 |
|---|---|
| 主入口 | `vad_lang_subtitle.py` |
| 输入 | FFmpeg 支持的视频/音频 |
| 输出 | `*.wav`, `*.srt`, `*.json` |
| 核心流程 | VAD -> Whisper -> Lingua -> 细化 |
| 典型用途 | 混合语言字幕生成 |

---

## 🚀 核心特性

- **Silero VAD -> Whisper 流水线**  
  先用语音活动检测（VAD）把音频切成语音片段，再由 Whisper 对每个片段转写。

- **细粒度语言检测**  
  结合 [Lingua](https://github.com/pemistahl/lingua-java) 与 Whisper 自带检测器，为每个片段（甚至单词）标注 ISO 语言代码（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）。

- **智能分段细化**  
  时间戳清洗可避免空隙或重叠；标点切分会在逗号、句号、问号等位置拆分过长文本；VAD 合并会将词重新对齐到 VAD 块以获得更顺滑字幕；长度感知分段会按语言应用不同限制。

- **多语言字幕输出**  
  同时输出 `.srt` 与 `.json`，并保留每段语言标签，便于你在下游播放器或编辑器中按语言做样式化或过滤。

- **稳健的媒体处理**  
  通过 FFmpeg 自动提取并归一化音频，尝试修复损坏容器，并应用动态归一化（`dynaudnorm`）以提升转写清晰度。

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
2. 根据输入文件名生成输出路径。
3. 通过 FFmpeg 提取并归一化音频。
4. 加载 Silero VAD（`torch.hub`）与 Whisper 模型。
5. 对 VAD 分块做第一轮转写。
6. 合并/细化分段后，对合并区间执行第二轮转写。
7. 执行字幕长度压缩与时间戳清理。
8. 保存 `.srt` 与 `.json`。

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

> ⚠️ 注意：此前 README 提到 `requirements.txt`，但当前仓库根目录中并不存在该文件。

---

## ✅ 前置要求

- Python `3.10+`（已在现代 3.x 环境测试）
- 已安装并可在 `PATH` 中访问 `ffmpeg`
- 依据 Whisper 模型选择，具备足够的 CPU/GPU 与内存（使用 `large` 时强烈建议 GPU）
- 首次运行需要联网下载 Whisper 模型权重与 Silero VAD 资源（`torch.hub`）

脚本使用到的 Python 包包括：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python 包）
- `lingua-language-detector`
- `tqdm`

快速校验命令：

```bash
python --version
ffmpeg -version
```

---

## 🔧 安装

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

如果你的代码副本中仍缺少 `requirements.txt`，可手动安装核心运行依赖：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

并确保系统层面已安装 FFmpeg。

---

## ⚡ 快速开始

如果你希望从克隆到生成字幕走最短路径：

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

提示：迭代阶段用 `small` 更快，最终产出可切换到 `large` 以提高质量。

在输入媒体旁边预期会生成：

- `*.wav` 提取并归一化后的音频
- `*.srt` 供播放器/编辑器使用的字幕文件
- `*.json` 结构化多语言字幕元数据

---

## 🛠 用法

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI 选项

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | 输入媒体路径（FFmpeg 支持视频/音频） |
| `--whisper-model` | — | No | Whisper 模型名（默认：`large`） |
| `--force` | — | No | 即使 `.wav`、`.srt`、`.json` 已存在也强制重跑 |

### 处理行为

- 输出文件名由输入基础路径推导。
- 对于 `input.mp4`，输出为 `input.wav`（归一化音频）、`input.srt`（带时间戳字幕）和 `input.json`（包含 `start`、`end`、`lang`、`text` 及可选词级时间信息的元数据）。
- 若已有 `.srt` 或 `.json`，默认会跳过；设置 `--force` 可强制覆盖。

---

## ⚙️ 配置

当前配置主要由 CLI 参数与代码默认值决定：

| 配置项 | 当前行为 |
|---|---|
| Whisper 模型 | `--whisper-model`（默认 `large`） |
| 处理采样率 | VAD/转写处理固定为 `16000` |
| FFmpeg 提取 | 单声道 WAV，`44100 Hz`，并使用 `dynaudnorm=f=100` |
| Lingua 检测器 | 主流程中初始化为 `ENGLISH`、`CHINESE`、`JAPANESE`、`ARABIC` |
| Whisper 侧过滤辅助默认值 | 包含 `en`、`zh`、`ja`、`ar`、`yue`、`ko`、`vi`、`es`、`fr` |

假设说明：辅助默认语言列表与主检测器设置并不完全一致；本 README 按当前实现保持一致描述。

---

## 📦 输出格式

该工具会为每个输入媒体写出两类字幕产物：

- `*.srt`：标准字幕文本，时间戳格式为 `HH:MM:SS,mmm`。
- `*.json`：结构化字幕列表，包含格式化时间戳与语言标签。

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

- `start`/`end` 在 JSON 输出中序列化为 SRT 风格字符串。
- `words` 是否存在取决于分段处理/细化阶段。
- 当语言不确定时，可能出现 `lang` 为 `und`。

---

## 🧪 示例

对 MP4 运行：

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

对 MOV 运行并强制覆盖：

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

对 FFmpeg 支持的纯音频输入运行：

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

- 当前规范的主脚本是 `vad_lang_subtitle.py`。
- 历史文件（`*.old`、`*.shorterlength*`、`archived/`）可用于参考，但通常不作为主线代码。
- 当前仓库未提交打包工程脚手架（`pyproject.toml`、`setup.py`），也没有 CI/测试套件。
- `data/` 含有较大的样例媒体；实验时请关注仓库体积与本地磁盘占用。
- 代码中的 `clean_subtitles_dict()` 存在，但当前主流水线未调用。
- `--force` 是当前保证重新生成输出、便于迭代调参的机制。

建议本地开发循环：

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

迭代时优先使用更小模型（`tiny`/`base`/`small`），最终输出再切换到 `large`。

---

## 🩺 故障排查

| 症状 | 处理建议 |
|---|---|
| `ffmpeg: command not found` | 安装 FFmpeg，并通过 `ffmpeg -version` 验证。 |
| 首次运行很慢或看似卡住 | 首次下载模型（Whisper + Silero）需要时间；后续重跑会更快。 |
| CUDA / GPU 错误 | 尝试改用更小 Whisper 模型（`small`、`base`、`tiny`）走 CPU 回退，并确保 PyTorch 构建与你的环境匹配。 |
| 输出文件没有重新生成 | 使用 `--force` 覆盖已有衍生文件。 |
| `pip install -r requirements.txt` 因文件缺失而失败 | 使用安装章节中的手动依赖安装命令。 |
| 短分段语言标注不准确 | 在极短/噪声音段上可能出现；当前逻辑结合了 Whisper 和 Lingua，但仍有边界场景。 |
| 字幕输出为空或接近为空 | 确认输入有语音，检查提取的 `.wav`，验证 FFmpeg 提取后加 `--force` 重试。 |
| 相邻字幕行语言频繁跳变 | 在很短分段上可能发生；可在下游工具里按语言和最小时长做后处理合并。 |

---

## ⚠️ 已知限制与假设

- 依赖清单尚未提交（编写时仓库根目录缺少 `requirements.txt`、`pyproject.toml`、`setup.py`）。
- README 声明许可为 MIT，但当前尚无独立 `LICENSE` 文件。
- 主流程中 Lingua 显式初始化为 `EN/ZH/JA/AR`，而辅助默认值包含更多候选语言代码。
- 当前没有已提交的自动化测试/基准，验证主要依赖手动流程。
- 根目录和 `archived/` 下保留了历史脚本；除非刻意实验，否则应以 `vad_lang_subtitle.py` 为准。

---

## 🗺 路线图

- 增加并维护固定版本的 `requirements.txt` 或 `pyproject.toml`。
- 为分段与时间戳清理逻辑增加自动化测试。
- 增加多语言边界场景的基准与质量评估文档。
- 增加可选配置文件支持，替代纯代码默认值配置。
- 扩展 `i18n/` 中的 README 多语言集合并保持语言导航同步。
- 统一检测器配置与辅助默认值之间的语言选择行为。
- 增加正式 `LICENSE` 文件以匹配 README 声明。

---

## 🔗 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) 提供语音转文本能力
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) 提供稳健的语音活动检测
- [Lingua](https://github.com/pemistahl/lingua-java) 提供高精度语言识别

---

## 🤝 贡献

1. Fork 并克隆仓库
2. 创建分支：`git checkout -b feat/your-idea`
3. 提交并推送
4. 发起 PR

若是较大改动，请附带：

- 预期行为变化的简短说明
- 可复现的命令示例
- 相关场景下字幕变更的前后对比片段

---

## 📄 许可

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
