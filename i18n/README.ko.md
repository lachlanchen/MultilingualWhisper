[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

OpenAI Whisper 기반의 즉시 사용 가능한 자막 생성기입니다. 혼합 언어가 포함된 영상에서도 세그먼트 단위 언어 감지와 정교한 보정을 통해 정확도를 높였습니다.

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

`MultilingualWhisper`는 [`vad_lang_subtitle.py`](vad_lang_subtitle.py)를 중심으로 구성된 Python CLI 파이프라인입니다. 다음 구성요소를 결합합니다.

- 음성 구간 분할을 위한 Silero VAD
- 전사 및 1차 언어 예측을 위한 OpenAI Whisper
- 텍스트 기반 언어 보정을 위한 Lingua
- 추출, 정규화, 미디어 처리를 위한 FFmpeg

주요 출력물은 `.srt`, `.json` 자막 파일과 정규화된 `.wav` 오디오입니다.

### At a Glance

| Item | Details |
|---|---|
| Main entrypoint | `vad_lang_subtitle.py` |
| Input | FFmpeg가 지원하는 비디오/오디오 |
| Output | `*.wav`, `*.srt`, `*.json` |
| Core flow | VAD -> Whisper -> Lingua -> refinement |
| Typical use case | 혼합 언어 자막 생성 |

---

## 🚀 Key Features

- **Silero VAD -> Whisper pipeline**
  Voice Activity Detection(VAD)로 오디오를 음성 세그먼트로 나눈 뒤, Whisper가 각 청크를 전사합니다.

- **Fine-grained language detection**
  [Lingua](https://github.com/pemistahl/lingua-java)와 Whisper 자체 감지 결과를 함께 사용하여 각 세그먼트(심지어 개별 단어 수준까지)에 ISO 언어 코드(`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...)를 태깅합니다.

- **Intelligent segment refinement**
  타임스탬프 정리를 통해 공백과 겹침을 제거합니다. 쉼표, 마침표, 물음표 등의 문장부호 기준 분할로 긴 전사를 나누고, VAD 병합으로 단어를 VAD 블록에 다시 정렬해 자막 흐름을 매끄럽게 만듭니다. 길이 인지 세분화는 언어별 제한을 적용합니다.

- **Multilingual subtitles**
  `.srt`와 `.json`을 모두 출력하며, 세그먼트별 언어 태그를 유지하므로 후속 플레이어나 편집기에서 언어별 스타일링/필터링이 가능합니다.

- **Robust media handling**
  FFmpeg를 통해 오디오를 자동 추출/정규화하고, 손상된 컨테이너 복구를 시도하며, 더 선명한 전사를 위해 동적 정규화(`dynaudnorm`)를 적용합니다.

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

`vad_lang_subtitle.py`의 주요 런타임 경로:

1. CLI 인자(`--video-path`, `--whisper-model`, `--force`)를 파싱합니다.
2. 입력 basename에서 출력 경로를 결정합니다.
3. FFmpeg로 오디오를 추출/정규화합니다.
4. Silero VAD(`torch.hub`)와 Whisper 모델을 로드합니다.
5. VAD 청크에 대해 1차 전사를 수행합니다.
6. 세그먼트를 병합/보정한 뒤 병합된 구간에 대해 2차 전사를 수행합니다.
7. 자막 길이 축소 및 타임스탬프 정리를 적용합니다.
8. `.srt`와 `.json`을 저장합니다.

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

> ⚠️ 참고: 이전 README에서는 `requirements.txt`를 언급했지만, 현재 저장소 루트에는 해당 파일이 없습니다.

---

## ✅ Prerequisites

- Python `3.10+` (최신 3.x 환경에서 테스트)
- `ffmpeg`가 설치되어 있고 `PATH`에서 접근 가능해야 함
- 선택한 Whisper 모델에 맞는 충분한 CPU/GPU + RAM (`large`는 GPU 권장)
- 첫 실행 시 Whisper 모델 가중치와 Silero VAD 자산(`torch.hub`) 다운로드를 위한 인터넷 연결

스크립트에서 사용하는 Python 패키지:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

빠른 확인 명령:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Installation

1. **이 저장소 클론**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **가상환경 생성 및 활성화**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **의존성 설치**

```bash
pip install -r requirements.txt
```

체크아웃에 `requirements.txt`가 아직 없다면, 핵심 런타임 의존성을 수동으로 설치하세요.

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

그리고 시스템 레벨에서 FFmpeg가 설치되어 있어야 합니다.

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
| `--video-path` | `-t` | Yes | 입력 미디어 경로(FFmpeg 지원 비디오/오디오) |
| `--whisper-model` | — | No | Whisper 모델 이름(기본값: `large`) |
| `--force` | — | No | `.wav`, `.srt`, `.json`이 이미 있어도 다시 실행 |

### Processing Behavior

- 출력 파일명은 입력 base path를 기준으로 생성됩니다.
- `input.mp4`라면 출력은 `input.wav`(정규화 오디오), `input.srt`(타임스탬프 자막), `input.json`(`start`, `end`, `lang`, `text`, 선택적으로 단어 타이밍 포함)입니다.
- `--force`를 지정하지 않으면 기존 `.srt` 또는 `.json`이 있을 때 건너뜁니다.

---

## ⚙️ Configuration

현재 설정은 주로 CLI 인자와 코드 기본값으로 제어됩니다.

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | VAD/전사용 처리 샘플레이트는 `16000`으로 하드코딩 |
| FFmpeg extraction | 모노 WAV, `44100 Hz`, `dynaudnorm=f=100` 적용 |
| Lingua detector | 메인 플로우에서 `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC`으로 초기화 |
| Whisper-side filtering helper defaults | `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` 포함 |

가정 참고: helper 기본값의 언어 목록과 메인 detector 설정은 완전히 동일하지 않습니다. 이 README는 구현된 현재 동작을 그대로 반영합니다.

---

## 📦 Output Format

도구는 입력 미디어마다 두 가지 자막 아티팩트를 생성합니다.

- `*.srt`: `HH:MM:SS,mmm` 타임스탬프를 사용하는 표준 자막 텍스트
- `*.json`: 포맷된 타임스탬프와 언어 태그를 담은 구조화된 자막 목록

일반적인 JSON 세그먼트 형태:

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

참고:

- JSON 출력의 `start`/`end`는 SRT 스타일 문자열로 직렬화됩니다.
- `words`는 세그먼트 처리/보정 단계에 따라 포함될 수 있습니다.
- 언어를 확신할 수 없는 구간에는 `lang` 값으로 `und`가 나타날 수 있습니다.

---

## 🧪 Examples

MP4에 실행:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV에 실행하고 강제 덮어쓰기:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

FFmpeg가 지원하는 오디오 전용 입력에 실행:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

배치 셸 예시(bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Development Notes

- 정식 활성 스크립트는 `vad_lang_subtitle.py`입니다.
- 이력 파일(`*.old`, `*.shorterlength*`, `archived/`)은 참고용으로 유용하지만 정식 경로는 아닌 것으로 보입니다.
- 현재 패키지 프로젝트 골격(`pyproject.toml`, `setup.py`)과 CI/테스트 스위트는 커밋되어 있지 않습니다.
- `data/`에는 대용량 샘플 미디어 아티팩트가 포함되어 있으므로 실험 시 저장소 크기와 로컬 디스크 사용량에 유의하세요.
- 코드에 `clean_subtitles_dict()`가 존재하지만 현재 메인 파이프라인에서 호출되지는 않습니다.
- 반복 튜닝 중 출력 재생성을 보장하는 현재 메커니즘은 `--force`입니다.

권장 로컬 개발 루프:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

반복 작업 중에는 작은 모델(`tiny`/`base`/`small`)을 사용하고, 최종 품질 산출 시 `large`로 전환하세요.

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | FFmpeg를 설치하고 `ffmpeg -version`으로 확인하세요. |
| First run is very slow or appears stuck | 초기 모델 다운로드(Whisper + Silero)에 시간이 걸릴 수 있으며, 재실행은 더 빠릅니다. |
| CUDA / GPU errors | 더 작은 Whisper 모델(`small`, `base`, `tiny`)로 CPU fallback을 시도하고, 환경에 맞는 PyTorch 빌드인지 확인하세요. |
| Output files are not regenerated | 기존 파생 파일을 덮어쓰려면 `--force`를 사용하세요. |
| `pip install -r requirements.txt` fails because file not found | Installation 섹션의 수동 의존성 설치 명령을 사용하세요. |
| Inaccurate language tagging on short segments | 매우 짧거나 노이즈가 큰 구간에서 발생할 수 있습니다. 현재 로직은 Whisper와 Lingua를 결합하지만 엣지 케이스가 남아 있습니다. |
| Empty or near-empty subtitle output | 입력에 음성이 있는지 확인하고, 추출된 `.wav`를 점검한 뒤 FFmpeg 추출이 정상인지 확인 후 `--force`로 재시도하세요. |
| Unexpected language flips between neighboring lines | 매우 짧은 세그먼트에서 발생할 수 있습니다. 후속 도구에서 언어 및 최소 길이 기준으로 post-merge를 고려하세요. |

---

## ⚠️ Known Limitations and Assumptions

- 의존성 매니페스트가 커밋되어 있지 않습니다(작성 시점 기준 저장소 루트에 `requirements.txt`, `pyproject.toml`, `setup.py` 부재).
- README에는 라이선스가 MIT로 선언되어 있지만, 독립된 `LICENSE` 파일은 현재 없습니다.
- 메인 플로우의 Lingua는 `EN/ZH/JA/AR`로 명시 초기화되지만, helper 기본값에는 더 많은 후보 코드가 포함됩니다.
- 자동화된 테스트/벤치마크가 아직 커밋되어 있지 않아 검증은 주로 수동으로 수행됩니다.
- 루트와 `archived/`에 이력 스크립트가 존재하며, 의도적으로 실험하지 않는 한 활성 스크립트는 `vad_lang_subtitle.py`로 간주해야 합니다.

---

## 🗺 Roadmap

- 고정 버전의 `requirements.txt` 또는 `pyproject.toml` 추가 및 유지
- 세그먼트 분할 및 타임스탬프 정리 로직에 대한 자동 테스트 추가
- 다국어 엣지 케이스 대상 벤치마크 및 품질 평가 문서 추가
- 코드 기본값 중심 동작 대신 선택적 설정 파일 지원 추가
- `i18n/`의 README 다국어 세트를 확장하고 language bar 동기화 유지
- detector 설정과 helper 기본값 간 언어 선택 동작을 명확화 및 통일
- README 선언과 일치하도록 정식 `LICENSE` 파일 추가

---

## 💖 Support

이 프로젝트가 도움이 되었다면 아래를 통해 개발을 후원할 수 있습니다.

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
2. 브랜치 생성: `git checkout -b feat/your-idea`
3. 커밋 후 push
4. PR 오픈

큰 변경의 경우 다음을 포함해 주세요.

- 예상 동작 변경에 대한 짧은 설명
- 재현 가능한 명령 예시
- 관련 있는 경우 자막 before/after 스니펫

---

## 📄 License

MIT © Lachlan Chen
