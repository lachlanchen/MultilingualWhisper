[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

OpenAI Whisper를 기반으로 한 드롭인 자막 생성기입니다. 여러 언어가 섞인 영상에서도 세그먼트 단위의 정밀한 언어 감지와 후처리 개선을 수행하도록 확장했습니다.

> 언어 인지 세그먼트 처리로 실제 혼합 언어 미디어에서 더 깔끔한 다국어 자막을 생성하세요.

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

> 🌍 **다국어 문서 제공**: 영어 + 10개 번역 README가 [`i18n/`](../i18n/)에 있으며, 상단 언어 바에서 바로 이동할 수 있습니다.

### Documentation Languages

| Locale | File |
| --- | --- |

| Focus | Value |
| --- | --- |
| Input | FFmpeg 호환 오디오/비디오 |
| Pipeline | VAD segmentation -> Whisper transcription -> Lingua refinement |
| Output | 정규화된 `*.wav`, `*.srt`, `*.json` |
| Best use | 세그먼트별 언어 태그가 필요한 혼합 언어 자막 |

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

`MultilingualWhisper`는 [`vad_lang_subtitle.py`](../vad_lang_subtitle.py)를 중심으로 동작하는 Python CLI 파이프라인입니다. 다음 구성 요소를 결합합니다.

- 음성 구간 분할용 Silero VAD
- 전사 및 1차 언어 예측용 OpenAI Whisper
- 텍스트 기반 언어 보정용 Lingua
- 추출, 정규화, 미디어 처리용 FFmpeg

주요 출력물은 `.srt`, `.json` 자막 파일과 추출/정규화된 `.wav` 오디오입니다.

### At a Glance

| Item | Details |
|---|---|
| Main entrypoint | `vad_lang_subtitle.py` |
| Input | FFmpeg가 지원하는 영상/오디오 |
| Output | `*.wav`, `*.srt`, `*.json` |
| Core flow | VAD -> Whisper -> Lingua -> refinement |
| Typical use case | 혼합 언어 자막 생성 |

---

## 🚀 Key Features

- **Silero VAD -> Whisper pipeline**  
  Voice Activity Detection(VAD)로 오디오를 발화 구간으로 분할한 뒤, Whisper가 각 청크를 전사합니다.

- **Fine-grained language detection**  
  [Lingua](https://github.com/pemistahl/lingua-java)와 Whisper 내장 감지를 함께 사용해 모든 세그먼트(단어 수준 포함)에 ISO 언어 코드(`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...)를 부여합니다.

- **Intelligent segment refinement**  
  타임스탬프 정리를 통해 구간 간 빈틈과 겹침을 제거합니다. 쉼표, 마침표, 물음표 등을 기준으로 긴 전사를 분할하며, VAD 병합 단계에서 단어를 VAD 블록에 재정렬해 자막 흐름을 부드럽게 만듭니다. 또한 언어별 길이 제한을 적용합니다.

- **Multilingual subtitles**  
  `.srt`와 `.json`을 모두 출력하고, 세그먼트별 언어 태그를 유지합니다. 따라서 후속 플레이어나 편집기에서 언어별 스타일링/필터링이 가능합니다.

- **Robust media handling**  
  FFmpeg로 오디오를 자동 추출 및 정규화하고, 손상된 컨테이너 복구를 시도하며, `dynaudnorm` 동적 정규화를 적용해 전사 가독성을 높입니다.

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

`vad_lang_subtitle.py`의 주요 실행 경로:

1. CLI 인자 파싱 (`--video-path`, `--whisper-model`, `--force`)
2. 입력 파일 basename 기준으로 출력 경로 결정
3. FFmpeg로 오디오 추출/정규화
4. Silero VAD(`torch.hub`) 및 Whisper 모델 로드
5. VAD 청크에 대해 1차 전사 수행
6. 세그먼트 병합/정제 후, 병합 구간에 대해 2차 전사 수행
7. 자막 길이 축소 및 타임스탬프 정리 적용
8. `.srt` 및 `.json` 저장

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

> ⚠️ 참고: 이전 README에는 `requirements.txt`가 언급되어 있었지만, 현재 저장소 루트에는 해당 파일이 없습니다.

---

## ✅ Prerequisites

- Python `3.10+` (최신 3.x 환경에서 테스트)
- `ffmpeg` 설치 및 `PATH` 등록
- 선택한 Whisper 모델을 실행할 충분한 CPU/GPU + RAM (`large`는 GPU 권장)
- 최초 실행 시 Whisper 가중치 및 Silero VAD 자산(`torch.hub`)을 내려받기 위한 인터넷 연결

스크립트에서 사용하는 Python 패키지:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python 패키지)
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
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
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

체크아웃에 `requirements.txt`가 여전히 없다면, 핵심 런타임 의존성을 수동 설치하세요.

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

그리고 시스템 수준에서 FFmpeg가 설치되어 있어야 합니다.

---

## ⚡ Quick Start

클론 직후 가장 빠르게 자막을 생성하려면:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

팁: 반복 작업 중에는 `small`을 사용하고, 최종 결과물은 `large`로 재실행하세요.

입력 미디어 옆에 생성되는 산출물:

- 정규화 추출 오디오 `*.wav`
- 플레이어/편집기용 자막 파일 `*.srt`
- 구조화된 다국어 자막 메타데이터 `*.json`

---

## 🎚 Model Selection Guide

속도와 품질 목표에 맞게 Whisper 모델을 선택하세요.

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | Fastest | Lowest | 빠른 스모크 테스트 및 파이프라인 검증 |
| `small` | Fast | Good | 일상 반복 개발 및 로컬 작업 |
| `medium` | Medium | Better | 균형 잡힌 프로덕션 워크플로 |
| `large` (default) | Slowest | Best | 최고 품질의 최종 자막 출력 |

실무 패턴:

1. `small --force`로 반복
2. 타이밍과 언어 태그 검증
3. 배포용 출력은 `large --force`로 재실행

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
| `--video-path` | `-t` | Yes | 입력 미디어 경로(FFmpeg가 지원하는 영상/오디오) |
| `--whisper-model` | — | No | Whisper 모델 이름(기본값: `large`) |
| `--force` | — | No | `.wav`, `.srt`, `.json`이 이미 있어도 강제 재실행 |

### Processing Behavior

- 출력 파일명은 입력의 기본 경로를 기준으로 생성됩니다.
- `input.mp4`라면 출력은 `input.wav`(정규화 오디오), `input.srt`(타임스탬프 자막), `input.json`(`start`, `end`, `lang`, `text`, 선택적 단어 타이밍 메타데이터 포함)입니다.
- 기존 `.srt` 또는 `.json`이 있으면 `--force` 없이는 건너뜁니다.

---

## ⚙️ Configuration

현재 설정은 주로 CLI 인자와 코드 기본값으로 제어됩니다.

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (기본 `large`) |
| Processing sample rate | VAD/전사 처리용으로 `16000` 하드코딩 |
| FFmpeg extraction | Mono WAV, `44100 Hz`, `dynaudnorm=f=100` 적용 |
| Lingua detector | 메인 플로우에서 `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC`로 초기화 |
| Whisper-side filtering helper defaults | `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` 포함 |

가정 메모: 헬퍼 기본 언어 목록과 메인 감지기 설정 목록은 완전히 동일하지 않으며, 이 README는 현재 구현 동작을 그대로 반영합니다.

현재 스크립트 기준 추가 구현 세부사항:

- 런타임에서 `torch.set_num_threads(1)`을 적용합니다.
- VAD 모델은 `torch.hub.load(...)`로 `snakers4/silero-vad`에서 로드합니다.
- 세그먼트 정리 시 언어가 `und`이거나 텍스트가 비어 있는 항목은 제거합니다.

---

## 📦 Output Format

도구는 입력 미디어마다 두 가지 자막 산출물을 생성합니다.

- `*.srt`: `HH:MM:SS,mmm` 타임스탬프를 갖는 표준 자막 텍스트
- `*.json`: 포맷된 타임스탬프와 언어 태그를 포함한 구조화 자막 목록

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
- `words` 필드는 세그먼트 처리/정제 단계에 따라 포함될 수 있습니다.
- 언어 불확실 구간에는 `lang` 값이 `und`로 나타날 수 있습니다.

---

## 🧪 Examples

MP4에서 실행:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV에서 실행하고 강제 덮어쓰기:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

FFmpeg가 지원하는 오디오 전용 입력에서 실행:

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

- 현재 정식 스크립트는 `vad_lang_subtitle.py`입니다.
- 이력 파일(`*.old`, `*.shorterlength*`, `archived/`)은 참고용으로 유용하지만 정식 경로는 아닙니다.
- 현재 `pyproject.toml`, `setup.py` 같은 패키징 스캐폴딩과 CI/테스트 스위트가 커밋되어 있지 않습니다.
- `data/`에는 큰 샘플 미디어 아티팩트가 포함되어 있으므로 실험 시 저장소 크기와 로컬 디스크 사용량에 유의하세요.
- 코드에 `clean_subtitles_dict()`가 존재하지만 현재 메인 파이프라인에서 호출되지는 않습니다.
- 반복 튜닝 중 출력 재생성을 보장하는 현재 메커니즘은 `--force`입니다.

권장 로컬 개발 루프:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

반복 중에는 작은 모델(`tiny`/`base`/`small`)을 사용하고, 최종 품질 출력 시 `large`로 전환하세요.

---

## 🩺 Troubleshooting

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | FFmpeg를 설치하고 `ffmpeg -version`으로 확인하세요. |
| First run is very slow or appears stuck | 최초 모델 다운로드(Whisper + Silero)에 시간이 걸릴 수 있으며 재실행은 더 빠릅니다. |
| CUDA / GPU errors | 더 작은 Whisper 모델(`small`, `base`, `tiny`)로 CPU fallback을 시도하고, 환경에 맞는 PyTorch 빌드를 확인하세요. |
| Output files are not regenerated | 기존 산출물 덮어쓰기를 위해 `--force`를 사용하세요. |
| `pip install -r requirements.txt` fails because file not found | Installation 섹션의 수동 의존성 설치 명령을 사용하세요. |
| Inaccurate language tagging on short segments | 매우 짧거나 잡음이 심한 구간에서 발생할 수 있으며, 현재 로직은 Whisper와 Lingua를 결합하지만 엣지 케이스가 남아 있습니다. |
| Empty or near-empty subtitle output | 입력에 음성이 있는지 확인하고, 추출된 `.wav`를 점검한 뒤 FFmpeg 추출이 정상인지 확인 후 `--force`로 재실행하세요. |
| Unexpected language flips between neighboring lines | 매우 짧은 세그먼트에서 발생할 수 있으며, 후처리 도구에서 언어와 최소 길이 기준으로 병합을 고려하세요. |
| FFmpeg extraction fails on damaged media | 스크립트가 컨테이너 복구(`-c copy -movflags +faststart`) 후 재시도하지만, 심하게 손상된 파일은 실패할 수 있습니다. |

빠른 진단:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Known Limitations and Assumptions

- 의존성 매니페스트(`requirements.txt`, `pyproject.toml`, `setup.py`)가 작성 시점 기준 저장소 루트에 커밋되어 있지 않습니다.
- README에는 MIT 라이선스로 표기되어 있지만, 독립된 `LICENSE` 파일은 현재 없습니다.
- 메인 플로우의 Lingua 초기화는 `EN/ZH/JA/AR`로 명시되어 있고, 헬퍼 기본값은 더 많은 후보 코드를 포함합니다.
- 자동화 테스트/벤치마크가 현재 커밋되어 있지 않아 검증은 주로 수동으로 이뤄집니다.
- 루트와 `archived/`에 이력 스크립트가 존재하며, 의도적으로 실험하지 않는 한 `vad_lang_subtitle.py`만 활성 스크립트로 취급해야 합니다.
- 현재 스크립트는 자세한 런타임 로그와 세그먼트별 디버그 출력을 표시하며, 이는 현 구현에서 정상 동작입니다.

---

## 🗺 Roadmap

- 고정 버전 `requirements.txt` 또는 `pyproject.toml` 추가 및 유지
- 세그먼트 분할 및 타임스탬프 정리 로직 자동 테스트 추가
- 다국어 엣지 케이스용 벤치마크 및 품질 평가 문서 추가
- 코드 기본값 중심 동작 대신 선택적 설정 파일 지원 추가
- `i18n/` README 세트를 확장하고 언어 바를 동기화 유지
- 감지기 설정과 헬퍼 기본값 간 언어 선택 동작을 명확화 및 통일
- README 선언과 일치하도록 정식 `LICENSE` 파일 추가

---

## 🔗 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) for robust voice activity detection
- [Lingua](https://github.com/pemistahl/lingua-java) for high-accuracy language identification

---

## 🤝 Contributing

1. 포크 후 클론
2. 브랜치 생성: `git checkout -b feat/your-idea`
3. 커밋 및 푸시
4. PR 오픈

규모가 큰 변경의 경우 다음을 포함해 주세요.

- 기대 동작 변경에 대한 짧은 설명
- 재현 가능한 명령 예시
- 필요 시 변경 전/후 자막 스니펫

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- 버그 리포트, 사용 문의, 기능 요청은 이슈를 등록해 주세요.
- 후원/기부 관련 문의는 위 Support 옵션을 이용해 주세요.

---

## 📄 License

MIT © Lachlan Chen
