[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

혼합 언어가 포함된 영상에 대해, 세그먼트 단위의 정밀 언어 감지와 보정을 확장한 OpenAI Whisper 기반 드롭인 자막 생성기입니다.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ 개요

`MultilingualWhisper`는 [`vad_lang_subtitle.py`](vad_lang_subtitle.py)를 중심으로 하는 Python CLI 파이프라인입니다. 다음을 결합합니다.

- 음성 구간 분할용 Silero VAD
- 전사용 OpenAI Whisper 및 1차 언어 예측
- 텍스트 기반 언어 보정용 Lingua
- 추출, 정규화, 미디어 처리를 위한 FFmpeg

주요 출력물은 `.srt`, `.json` 자막 파일과, 추출/정규화된 `.wav` 오디오입니다.

### 한눈에 보기

| 항목 | 세부 내용 |
|---|---|
| 메인 엔트리포인트 | `vad_lang_subtitle.py` |
| 입력 | FFmpeg가 지원하는 비디오/오디오 |
| 출력 | `*.wav`, `*.srt`, `*.json` |
| 핵심 흐름 | VAD -> Whisper -> Lingua -> refinement |
| 대표 사용 사례 | 혼합 언어 자막 생성 |

---

## 🚀 주요 기능

- **Silero VAD -> Whisper 파이프라인**  
  음성 활동 감지(VAD)로 오디오를 발화 구간으로 분할한 뒤, Whisper가 각 청크를 전사합니다.

- **세밀한 언어 감지**  
  [Lingua](https://github.com/pemistahl/lingua-java)를 Whisper 자체 감지와 함께 사용해, 모든 세그먼트(개별 단어 포함)에 ISO 언어 코드(`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...)를 태깅합니다.

- **지능형 세그먼트 보정**  
  타임스탬프 정리로 공백/겹침을 방지합니다. 쉼표, 마침표, 물음표 등 문장부호 기준 분할로 긴 전사문을 나눕니다. VAD 병합 재정렬로 단어를 VAD 블록에 다시 맞춰 자막 흐름을 부드럽게 합니다. 언어별 길이 제한을 적용한 분할도 포함됩니다.

- **다국어 자막 출력**  
  `.srt`와 `.json`을 모두 출력하며, 세그먼트별 언어 태그를 유지해 후속 플레이어/에디터에서 언어별 스타일링 또는 필터링이 가능합니다.

- **견고한 미디어 처리**  
  FFmpeg로 오디오를 자동 추출/정규화하고, 손상된 컨테이너 복구를 시도하며, 더 선명한 전사를 위해 동적 정규화(`dynaudnorm`)를 적용합니다.

---

## 🗂 프로젝트 구조

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

> ⚠️ 참고: 이전 README에서 `requirements.txt`를 언급했지만, 현재 저장소 루트에는 파일이 없습니다.

---

## ✅ 사전 요구사항

- Python `3.10+` (최신 3.x 환경에서 테스트)
- `PATH`에서 접근 가능한 `ffmpeg` 설치
- 선택한 Whisper 모델을 실행할 충분한 CPU/GPU + RAM (`large`는 GPU 강력 권장)
- 첫 실행 시 Whisper 모델 가중치와 Silero VAD 리소스(`torch.hub`)를 가져오기 위한 인터넷 연결

스크립트에서 사용하는 Python 패키지:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python 패키지)
- `lingua-language-detector`
- `tqdm`

---

## 🔧 설치

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

체크아웃에 `requirements.txt`가 여전히 없다면, 핵심 런타임 의존성을 수동으로 설치하세요.

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

그리고 시스템 레벨에서 FFmpeg가 설치되어 있어야 합니다.

---

## 🛠 사용법

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### CLI 옵션

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | 입력 미디어 경로(FFmpeg 지원 비디오/오디오) |
| `--whisper-model` | — | No | Whisper 모델명 (기본값: `large`) |
| `--force` | — | No | `.wav`, `.srt`, `.json`이 이미 있어도 재실행 |

### 처리 동작

- 출력 파일명은 입력 파일의 베이스 경로에서 파생됩니다.
- `input.mp4` 기준 출력은 `input.wav`(정규화 오디오), `input.srt`(타임스탬프 자막), `input.json`(`start`, `end`, `lang`, `text`, 선택적으로 단어 타이밍 포함 메타데이터)입니다.
- `--force`를 지정하지 않으면, 기존 `.srt` 또는 `.json`이 있을 때 건너뜁니다.

---

## ⚙️ 설정

현재 설정은 주로 CLI 인자와 코드 기본값에 의해 결정됩니다.

- Whisper 모델: `--whisper-model` (기본값 `large`)
- 샘플링 레이트: 처리 시 `16000`으로 하드코딩
- FFmpeg 추출: 모노 WAV, `44100 Hz`, `dynaudnorm=f=100` 적용
- Lingua 감지기: 메인 흐름에서 `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC`으로 초기화
- Whisper 측 필터링용 허용 언어 코드는 헬퍼 기본값 기준 `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` 포함

가정 참고: 헬퍼 기본값의 언어 목록과 메인 감지기 설정은 완전히 동일하지 않습니다. 이 README는 구현된 현재 동작을 보존해 설명합니다.

---

## 🧪 예시

MP4 실행:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

MOV 실행 + 강제 덮어쓰기:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

FFmpeg가 지원하는 오디오 전용 입력 실행:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 개발 노트

- 기준이 되는 활성 스크립트는 `vad_lang_subtitle.py`입니다.
- 히스토리 파일(`*.old`, `*.shorterlength*`, `archived/`)은 참고용으로 유용하지만 비표준(canonical)로 보입니다.
- 현재 패키징 프로젝트 스캐폴딩(`pyproject.toml`, `setup.py`)과 CI/테스트 스위트가 커밋되어 있지 않습니다.
- `data/`에는 대용량 샘플 미디어 산출물이 포함되어 있으므로, 실험 시 저장소 크기와 로컬 디스크 사용량에 유의하세요.
- 코드에 `clean_subtitles_dict()`가 존재하지만 메인 파이프라인에서는 현재 호출되지 않습니다.

---

## 🩺 문제 해결

| 증상 | 조치 방법 |
|---|---|
| `ffmpeg: command not found` | FFmpeg를 설치하고 `ffmpeg -version`으로 확인하세요. |
| 첫 실행이 매우 느리거나 멈춘 것처럼 보임 | 초기 모델 다운로드(Whisper + Silero)에 시간이 걸릴 수 있으며, 재실행은 더 빠릅니다. |
| CUDA / GPU 오류 | 더 작은 Whisper 모델(`small`, `base`, `tiny`)로 CPU 폴백을 시도하고, 환경에 맞는 PyTorch 빌드인지 확인하세요. |
| 출력 파일이 재생성되지 않음 | 기존 파생 파일을 덮어쓰려면 `--force`를 사용하세요. |
| `requirements.txt` 미존재로 `pip install -r requirements.txt` 실패 | 설치 섹션의 수동 의존성 설치 명령을 사용하세요. |
| 짧은 세그먼트에서 언어 태깅이 부정확함 | 매우 짧거나 노이즈가 큰 구간에서 발생할 수 있습니다. 현재 로직은 Whisper+Lingua 결합이지만 엣지 케이스는 남아 있습니다. |

---

## 🗺 로드맵

- 버전 고정된 `requirements.txt` 또는 `pyproject.toml` 추가 및 유지
- 세그먼트 분할/타임스탬프 정리 로직 자동 테스트 추가
- 다국어 엣지 케이스에 대한 벤치마크 및 품질 평가 문서 추가
- 코드 기본값 중심 동작 대신 선택적 설정 파일 지원 추가
- `i18n/`의 README 언어 세트 확장 및 언어 바 동기화 유지

---

## 💖 후원

이 프로젝트가 도움이 되었다면, 아래를 통해 개발을 지원할 수 있습니다.

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 감사의 글

- [OpenAI Whisper](https://github.com/openai/whisper) (음성 인식)
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) (견고한 음성 활동 감지)
- [Lingua](https://github.com/pemistahl/lingua-java) (고정확도 언어 식별)

---

## 🤝 기여

1. Fork 후 clone
2. 브랜치 생성: `git checkout -b feat/your-idea`
3. 커밋 후 push
4. PR 생성

---

## 📄 라이선스

MIT © Lachlan Chen
