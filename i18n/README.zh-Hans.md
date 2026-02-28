[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

一个基于 OpenAI Whisper 的即插即用字幕生成器，针对包含混合语言的视频扩展了精细的逐片段语言检测与优化能力。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ 概览

`MultilingualWhisper` 是一个以 [`vad_lang_subtitle.py`](vad_lang_subtitle.py) 为核心的 Python CLI 流水线，结合了：

- 使用 Silero VAD 进行语音分段
- 使用 OpenAI Whisper 进行转写与初始语言预测
- 使用 Lingua 进行基于文本的语言细化
- 使用 FFmpeg 进行提取、归一化与媒体处理

主要输出为 `.srt` 与 `.json` 字幕文件，以及提取并归一化后的 `.wav` 音频。

### 快速一览

| 项目 | 详情 |
|---|---|
| 主入口 | `vad_lang_subtitle.py` |
| 输入 | FFmpeg 支持的视频/音频 |
| 输出 | `*.wav`, `*.srt`, `*.json` |
| 核心流程 | VAD -> Whisper -> Lingua -> refinement |
| 典型场景 | 混合语言字幕生成 |

---

## 🚀 核心特性

- **Silero VAD -> Whisper 流水线**  
  通过语音活动检测（VAD）将音频切分为语音片段，再由 Whisper 对每个片段进行转写。

- **细粒度语言检测**  
  结合 [Lingua](https://github.com/pemistahl/lingua-java) 与 Whisper 自带检测器，为每个片段（甚至单词级）打上 ISO 语言代码（`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...）。

- **智能片段优化**  
  时间戳清理可避免间隙或重叠；标点切分会在逗号、句号、问号等位置拆分长文本；VAD 合并会将词级结果重新对齐到 VAD 块以获得更平滑字幕；同时结合语言相关的长度限制进行分段。

- **多语言字幕输出**  
  同时输出 `.srt` 与 `.json`，并保留每段语言标签，便于在下游播放器或编辑器中按语言样式化或筛选。

- **稳健的媒体处理**  
  通过 FFmpeg 自动提取并归一化音频，尝试修复损坏容器，并应用动态归一化（`dynaudnorm`）以提升转写清晰度。

---

## 🗂 项目结构

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

> ⚠️ 注意：此前 README 提到过 `requirements.txt`，但它目前在仓库根目录中缺失。

---

## ✅ 前置要求

- Python `3.10+`（已在现代 3.x 环境中测试）
- 已安装 `ffmpeg` 且可在 `PATH` 中访问
- 根据所选 Whisper 模型具备足够 CPU/GPU 与内存（若使用 `large`，强烈建议 GPU）
- 首次运行需要联网下载 Whisper 模型权重与 Silero VAD 资源（`torch.hub`）

脚本使用的 Python 包包括：

- `torch`
- `torchaudio`
- `whisper`（OpenAI Whisper Python 包）
- `lingua-language-detector`
- `tqdm`

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

如果你的仓库副本中仍缺少 `requirements.txt`，请手动安装核心运行依赖：

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

并确保系统层面已安装 FFmpeg。

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
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### 处理行为

- 输出文件名由输入路径的基础名派生。
- 对于 `input.mp4`，输出为 `input.wav`（归一化音频）、`input.srt`（带时间戳字幕）和 `input.json`（包含 `start`、`end`、`lang`、`text`，以及可选词级时间信息的元数据）。
- 若已存在 `.srt` 或 `.json`，默认会跳过，除非设置 `--force`。

---

## ⚙️ 配置

当前配置主要由 CLI 参数与代码默认值驱动：

- Whisper 模型：`--whisper-model`（默认 `large`）
- 采样率：处理流程中硬编码为 `16000`
- FFmpeg 提取：单声道 WAV、`44100 Hz`，并使用 `dynaudnorm=f=100`
- Lingua 检测器：主流程中初始化为 `ENGLISH`、`CHINESE`、`JAPANESE`、`ARABIC`
- Whisper 侧过滤所允许的语言代码在辅助默认值中包含 `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`

假设说明：辅助默认值中的语言列表与主检测器设置并不完全一致；本 README 仅保留当前实现行为。

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

---

## 🧭 开发说明

- 当前规范主脚本为 `vad_lang_subtitle.py`。
- 历史文件（`*.old`、`*.shorterlength*`、`archived/`）可供参考，但看起来不是规范实现。
- 当前仓库未提交打包脚手架（`pyproject.toml`、`setup.py`）与 CI/测试套件。
- `data/` 含有较大的示例媒体产物；实验时请注意仓库体积与本地磁盘占用。
- `clean_subtitles_dict()` 在代码中存在，但当前主流程未调用。

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

---

## 🗺 路线图

- 添加并维护固定版本的 `requirements.txt` 或 `pyproject.toml`。
- 为分段与时间戳清理逻辑添加自动化测试。
- 为多语言边缘场景补充基准与质量评估文档。
- 添加可选配置文件支持，而不仅依赖代码默认值。
- 扩展 `i18n/` 中 README 语言集，并保持语言导航栏同步。

---

## 💖 支持

如果这个项目对你有帮助，可以通过以下方式支持开发：

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 致谢

- [OpenAI Whisper](https://github.com/openai/whisper)（语音转文本）
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models)（稳健的语音活动检测）
- [Lingua](https://github.com/pemistahl/lingua-java)（高准确率语言识别）

---

## 🤝 贡献

1. Fork 并克隆
2. 创建分支：`git checkout -b feat/your-idea`
3. 提交并推送
4. 发起 PR

---

## 📄 许可证

MIT © Lachlan Chen
