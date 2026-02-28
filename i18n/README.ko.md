[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

OpenAI Whisper 기반으로 만든 바로 사용 가능한 자막 생성기입니다. 혼합 언어가 들어간 영상에서 세그먼트 단위 언어 감지와 정제 로직을 적용해 더 정확한 결과를 제공합니다.

> 실제 멀티링구얼 미디어에서 언어 인지 분할로 더 깔끔한 다국어 자막을 빠르게 생성합니다.

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

| Focus | Value |
| --- | --- |
| Input | FFmpeg-compatible audio/video |
| Pipeline | VAD segmentation → Whisper transcription → Lingua refinement |
| Output | Normalized `*.wav`, `*.srt`, and `*.json` |
| Best use | Mixed-language subtitles with per-segment language tags |

---

## 목차

- [개요](#-개요)
- [한눈에 보기](#-한눈에-보기)
- [주요 기능](#-주요-기능)
- [파이프라인 흐름](#-파이프라인-흐름)
- [프로젝트 구조](#-프로젝트-구조)
- [사전 요구사항](#-사전-요구사항)
- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [사용법](#-사용법)
- [설정](#-설정)
- [출력 형식](#-출력-형식)
- [예시](#-예시)
- [개발 노트](#-개발-노트)
- [문제 해결](#-문제-해결)
- [알려진 제한사항 및 가정](#-알려진-제한사항-및-가정)
- [로드맵](#-로드맵)
- [Support](#-support)
- [감사의 말](#-감사의-말)
- [기여](#-기여)
- [Contact](#-contact)
- [License](#-license)

---

## ✨ 개요

`MultilingualWhisper`는 [`vad_lang_subtitle.py`](vad_lang_subtitle.py)를 중심으로 동작하는 Python CLI 파이프라인입니다. 다음 구성 요소를 결합합니다.

- Silero VAD로 발화 구간 분할
- OpenAI Whisper로 전사 및 1차 언어 예측
- Lingua로 텍스트 기반 언어 정제
- FFmpeg를 통한 추출, 정규화, 미디어 처리

주요 산출물은 `.srt` 및 `.json` 자막 파일과 추출된 정규화 `*.wav` 오디오입니다.

### 한눈에 보기

| 항목 | 상세 |
|---|---|
| 메인 엔트리포인트 | `vad_lang_subtitle.py` |
| 입력 | FFmpeg에서 지원하는 비디오/오디오 |
| 출력 | `*.wav`, `*.srt`, `*.json` |
| 핵심 흐름 | VAD -> Whisper -> Lingua -> 정제 |
| 대표 사용 사례 | 혼합 언어 자막 생성 |

---

## 🚀 주요 기능

- **Silero VAD -> Whisper 파이프라인**
  음성 활동 탐지(VAD)를 통해 오디오를 발화 구간으로 나눈 뒤, Whisper가 각 조각을 전사합니다.

- **세밀한 언어 감지**
  [Lingua](https://github.com/pemistahl/lingua-java)를 Whisper의 자체 감지 기능과 함께 사용해 각 세그먼트(개별 단어 수준까지)에 ISO 언어 코드(`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...)를 부여합니다.

- **지능형 세그먼트 정제**
  타임스탬프 정리로 공백/겹침을 제거하고, 쉼표·마침표·물음표 등 구두점에서 긴 전사를 분할합니다. VAD 병합은 단어를 VAD 블록으로 다시 정렬해 자막 흐름을 부드럽게 만듭니다. 길이 인지 분할은 언어별 제한을 적용합니다.

- **다국어 자막 출력**
  `.srt`와 `.json`을 모두 생성하며, 세그먼트별 언어 태그를 보존해 하위 플레이어나 편집기에서 언어 기반 스타일링/필터링이 가능합니다.

- **견고한 미디어 처리**
  FFmpeg로 오디오를 자동 추출·정규화하고, 손상된 컨테이너 복구를 시도하며, 더 선명한 전사를 위해 `dynaudnorm` 동적 정규화를 적용합니다.

---

## 🔁 파이프라인 흐름

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

1. CLI 인자(`--video-path`, `--whisper-model`, `--force`) 파싱.
2. 입력 파일 basename을 기준으로 출력 경로 결정.
3. FFmpeg로 오디오 추출/정규화.
4. Silero VAD(`torch.hub`) 및 Whisper 모델 로드.
5. VAD 청크에 대해 1차 전사 수행.
6. 병합/정제된 세그먼트에 대해 2차 전사 수행.
7. 자막 길이 축소 및 타임스탬프 정리 적용.
8. `.srt`, `.json` 저장.

---

## 🗂 프로젝트 구조

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

> ⚠️ 참고: 이전 README에서 `requirements.txt`를 언급했지만 현재 저장소 루트에는 해당 파일이 없습니다.

---

## ✅ 사전 요구사항

- Python `3.10+` (최신 3.x 환경에서 테스트됨)
- `ffmpeg`가 설치되어 있고 `PATH`에서 사용 가능해야 함
- 선택한 Whisper 모델에 맞는 충분한 CPU/GPU + RAM( `large`의 경우 GPU 강력 권장)
- 첫 실행 시 Whisper 모델 가중치와 Silero VAD 자산(`torch.hub`)을 내려받기 위해 인터넷 접속 필요

스크립트에서 사용하는 Python 패키지:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python 패키지)
- `lingua-language-detector`
- `tqdm`

빠른 확인 명령어:

```bash
python --version
ffmpeg -version
```

---

## 🔧 설치

1. **저장소 클론**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **가상 환경 생성 및 활성화**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **의존성 설치**

```bash
pip install -r requirements.txt
```

체크아웃에서 `requirements.txt`가 여전히 없으면 핵심 런타임 의존성을 수동으로 설치하세요:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

그리고 시스템 수준에서 FFmpeg가 설치되어 있는지 확인하세요.

---

## ⚡ 빠른 시작

클론 후 가장 빠른 경로:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

팁: 반복 실험 시 `small`을 먼저 쓰고, 최종 품질이 중요할 때는 `large`로 전환하세요.

입력 미디어 옆에 생성되는 결과물:

- `*.wav`: 정규화된 추출 오디오
- `*.srt`: 플레이어/편집기용 자막 파일
- `*.json`: 구조화된 다국어 자막 메타데이터

---

## 🛠 사용법

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI 옵션

| 플래그 | 별칭 | 필수 | 설명 |
|---|---|---|---|
| `--video-path` | `-t` | Yes | 입력 미디어 경로 (FFmpeg가 지원하는 비디오/오디오) |
| `--whisper-model` | — | No | Whisper 모델 이름 (기본값: `large`) |
| `--force` | — | No | `.wav`, `.srt`, `.json`이 이미 있어도 다시 실행 |

### 처리 동작

- 출력 파일명은 입력 파일의 base path를 기준으로 생성됩니다.
- `input.mp4`의 경우 `input.wav`(정규화 오디오), `input.srt`(타임스탬프 자막), `input.json`(메타데이터 포함: `start`, `end`, `lang`, `text`, 필요 시 단어 타이밍)이 생성됩니다.
- `--force`를 지정하지 않으면 기존의 `.srt` 또는 `.json`이 있을 때 건너뜁니다.

---

## ⚙️ 설정

현재 설정은 주로 CLI와 코드 기본값으로 제어됩니다.

| 설정 항목 | 현재 동작 |
|---|---|
| Whisper model | `--whisper-model` (기본값 `large`) |
| 처리 샘플 레이트 | VAD/전사 처리 샘플레이트가 `16000`으로 하드코딩 |
| FFmpeg 추출 | 모노 WAV, `44100 Hz`, `dynaudnorm=f=100` |
| Lingua detector | 메인 흐름에서 `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC`로 초기화 |
| Whisper-side filtering helper defaults | `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` 포함 |

참고: helper 기본 언어 목록과 메인 detector 설정은 완전히 동일하지 않습니다. 이 README는 현재 구현된 동작을 그대로 반영합니다.

---

## 📦 출력 형식

도구는 입력 미디어마다 두 가지 자막 산출물을 저장합니다.

- `*.srt`: `HH:MM:SS,mmm` 타임스탬프 형식의 표준 자막 텍스트.
- `*.json`: 형식화된 타임스탬프와 언어 태그가 포함된 구조화된 자막 목록.

JSON 세그먼트 예시:

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

- `start`/`end`는 JSON 출력에서 SRT 스타일 문자열로 직렬화됩니다.
- `words`는 세그먼트 처리/정제 단계에 따라 포함될 수 있습니다.
- 불확실한 구간에서는 `und` 언어 코드가 나타날 수 있습니다.

---

## 🧪 예시

MP4 실행:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV에서 강제 덮어쓰기 실행:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

오디오-only(FFmpeg 지원) 입력 실행:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

배치 예시 (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 개발 노트

- 정식 메인 스크립트는 `vad_lang_subtitle.py`입니다.
- 과거 파일(`*.old`, `*.shorterlength*`, `archived/`)은 참고용으로 유용하지만, 공식 경로로 간주되지 않습니다.
- 현재 패키지 메타/빌드 구성(`pyproject.toml`, `setup.py`)과 CI/테스트 스위트가 커밋되어 있지 않습니다.
- `data/`에는 대용량 샘플 미디어가 있을 수 있으므로 실험 시 저장소 크기와 로컬 디스크 사용량에 유의하세요.
- 코드에 `clean_subtitles_dict()`가 존재하지만 현재 메인 파이프라인에서는 호출되지 않습니다.
- 반복 튜닝에서 산출물 재생성을 보장하는 현재 수단은 `--force`입니다.

권장 로컬 개발 루프:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

작은 모델(`tiny`/`base`/`small`)로 검증하고, 최종 산출에서는 `large`로 전환하세요.

---

## 🩺 문제 해결

| 증상 | 조치 |
|---|---|
| `ffmpeg: command not found` | FFmpeg를 설치하고 `ffmpeg -version`으로 확인하세요. |
| 첫 실행이 매우 느리거나 멈춘 것처럼 보임 | 초기 모델 다운로드(Whisper + Silero)에는 시간이 걸릴 수 있으며, 재실행은 더 빠릅니다. |
| CUDA / GPU 오류 | 더 작은 Whisper 모델(`small`, `base`, `tiny`)로 CPU 폴백을 시도하고, 환경에 맞는 PyTorch 빌드를 사용하세요. |
| 출력 파일이 다시 생성되지 않음 | 기존 파생 파일을 덮어쓰려면 `--force`를 사용하세요. |
| `pip install -r requirements.txt` 실행 시 파일 없음 오류 | 설치 섹션에 제시된 수동 의존성 설치 명령을 사용하세요. |
| 짧은 세그먼트에서 언어 태그가 부정확함 | 매우 짧거나 노이즈가 큰 구간에서 발생할 수 있습니다. 현재 로직은 Whisper와 Lingua를 결합하지만 경계 케이스가 남아 있습니다. |
| 자막이 비어 있거나 거의 비어 있음 | 입력에 실제 음성이 있는지 확인하고, 추출한 `.wav`를 점검한 후 FFmpeg 추출 결과를 확인하고 `--force`로 재시도하세요. |
| 인접 줄 간 언어 전환이 급격함 | 매우 짧은 세그먼트에서 발생할 수 있습니다. 후처리에서 언어별 그룹화 및 최소 지속시간 기준 병합을 고려하세요. |

---

## ⚠️ 알려진 제한사항 및 가정

- 의존성 매니페스트가 커밋되어 있지 않습니다 (`requirements.txt`, `pyproject.toml`, `setup.py`가 현재 루트에 없음).
- 라이선스는 README에 MIT로 기재되어 있으나, 별도의 `LICENSE` 파일이 현재 없습니다.
- Lingua는 메인 흐름에서 `EN/ZH/JA/AR`로 초기화하지만 helper 기본값에는 더 많은 코드가 포함되어 있습니다.
- 자동화된 테스트/벤치마크가 현재 없어 검증은 주로 수동으로 수행됩니다.
- 루트와 `archived/`에 과거 스크립트가 존재합니다. 의도적으로 실험하지 않는 한 정식은 `vad_lang_subtitle.py`만 사용하세요.

---

## 🗺 로드맵

- 고정 버전의 `requirements.txt` 또는 `pyproject.toml` 추가 및 유지
- 세그먼트 분할 및 타임스탬프 정리 로직에 대한 자동 테스트 추가
- 다국어 엣지 케이스의 벤치마크 및 품질 평가 문서화
- 코드 기본값 중심 동작 대신 선택적 설정 파일 지원 추가
- `i18n/`의 README 다국어 목록을 확장하고 언어 바를 동기화 상태로 유지
- detector 설정과 helper 기본값 간 언어 선택 동작 정합성 개선
- README의 선언과 일치하도록 정식 `LICENSE` 파일 추가

---

## 🔗 감사의 말

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) for robust voice activity detection
- [Lingua](https://github.com/pemistahl/lingua-java) for high-accuracy language identification

---

## 🤝 기여

1. Fork 및 clone
2. 브랜치 생성: `git checkout -b feat/your-idea`
3. 변경사항 커밋 및 push
4. PR 열기

주요 변경의 경우 다음을 포함하세요:

- 예상되는 동작 변경의 간단한 설명
- 재현 가능한 명령어 예시
- 가능할 경우 자막 before/after 스니펫

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- 버그 리포트, 사용법 문의, 기능 요청은 issue를 열어주세요.
- 후원/기부 문의는 위의 지원 항목을 이용하세요.

---

## 📄 License

MIT © Lachlan Chen
