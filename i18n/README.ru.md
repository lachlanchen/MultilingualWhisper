[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Генератор субтитров «подключил и используешь», построенный на OpenAI Whisper и дополненный точным определением языка по сегментам и их уточнением для видео со смешанной речью.

> Создавайте более чистые многоязычные субтитры из реального медиа со смешанными языками благодаря сегментации с учетом языка.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)

---

## Содержание

- [Обзор](#-обзор)
- [Кратко](#кратко)
- [Ключевые возможности](#-ключевые-возможности)
- [Пайплайн](#-пайплайн)
- [Структура проекта](#-структура-проекта)
- [Требования](#-требования)
- [Установка](#-установка)
- [Быстрый старт](#-быстрый-старт)
- [Использование](#-использование)
- [Конфигурация](#️-конфигурация)
- [Формат вывода](#-формат-вывода)
- [Примеры](#-примеры)
- [Заметки по разработке](#-заметки-по-разработке)
- [Устранение неполадок](#-устранение-неполадок)
- [Известные ограничения и допущения](#️-известные-ограничения-и-допущения)
- [Дорожная карта](#-дорожная-карта)
- [Support](#-support)
- [Благодарности](#-благодарности)
- [Участие в проекте](#-участие-в-проекте)
- [Лицензия](#-лицензия)

---

## ✨ Обзор

`MultilingualWhisper` — это Python CLI-пайплайн, сфокусированный на [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Он объединяет:

- Silero VAD для сегментации речи
- OpenAI Whisper для транскрибации и первичного определения языка
- Lingua для уточнения языка по тексту
- FFmpeg для извлечения, нормализации и обработки медиа

Основные выходные артефакты: файлы субтитров `.srt` и `.json`, а также извлеченное нормализованное аудио `.wav`.

### Кратко

| Пункт | Детали |
|---|---|
| Главная точка входа | `vad_lang_subtitle.py` |
| Вход | Видео/аудио, поддерживаемые FFmpeg |
| Выход | `*.wav`, `*.srt`, `*.json` |
| Базовый поток | VAD -> Whisper -> Lingua -> refinement |
| Типовой сценарий | Генерация субтитров для смешанных языков |

---

## 🚀 Ключевые возможности

- **Пайплайн Silero VAD -> Whisper**
  Voice Activity Detection (VAD) разбивает аудио на речевые сегменты, после чего Whisper транскрибирует каждый фрагмент.

- **Детальное определение языка**
  Используется [Lingua](https://github.com/pemistahl/lingua-java) вместе со встроенным детектором Whisper, чтобы помечать каждый сегмент (и даже отдельные слова) ISO-кодами языков (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Интеллектуальное уточнение сегментов**
  Очистка таймкодов обеспечивает отсутствие разрывов и наложений. Разбиение по пунктуации делит длинные транскрипции по запятым, точкам, вопросительным знакам и т.д. Объединение по VAD заново выравнивает слова к VAD-блокам для более плавных субтитров. Сегментация с учетом длины применяет языково-специфичные лимиты.

- **Многоязычные субтитры**
  Генерирует и `.srt`, и `.json`, сохраняя языковые метки на уровне сегментов, чтобы вы могли стилизовать или фильтровать субтитры по языку в downstream-плеерах или редакторах.

- **Надежная обработка медиа**
  Автоматически извлекает и нормализует аудио через FFmpeg, пытается исправлять поврежденные контейнеры и применяет динамическую нормализацию (`dynaudnorm`) для более четкой транскрибации.

---

## 🔁 Пайплайн

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

Основной runtime-путь в `vad_lang_subtitle.py`:

1. Разбор аргументов CLI (`--video-path`, `--whisper-model`, `--force`).
2. Определение путей выхода из basename входного файла.
3. Извлечение/нормализация аудио через FFmpeg.
4. Загрузка Silero VAD (`torch.hub`) и модели Whisper.
5. Первый проход транскрибации по VAD-фрагментам.
6. Объединение/уточнение сегментов, затем второй проход транскрибации на объединенных интервалах.
7. Применение сокращения длины субтитров и очистки таймкодов.
8. Сохранение `.srt` и `.json`.

---

## 🗂 Структура проекта

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

> ⚠️ Примечание: в предыдущем README упоминался `requirements.txt`, но сейчас он отсутствует в корне репозитория.

---

## ✅ Требования

- Python `3.10+` (проверено на современных окружениях 3.x)
- Установленный `ffmpeg`, доступный в `PATH`
- Достаточный CPU/GPU и RAM для выбранной модели Whisper (для `large` настоятельно рекомендуется GPU)
- Интернет при первом запуске для загрузки весов Whisper и ассетов Silero VAD (`torch.hub`)

Python-пакеты, используемые скриптом:

- `torch`
- `torchaudio`
- `whisper` (Python-пакет OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

Команды быстрой проверки:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Установка

1. **Клонируйте репозиторий**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **Создайте и активируйте виртуальное окружение**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Установите зависимости**

```bash
pip install -r requirements.txt
```

Если `requirements.txt` в вашей копии все еще отсутствует, установите основные runtime-зависимости вручную:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

И убедитесь, что FFmpeg установлен на уровне системы.

---

## ⚡ Быстрый старт

Если нужен самый быстрый путь от клона до субтитров:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Совет: используйте `small` во время итераций, затем переключайтесь на `large` для финального качества.

Ожидаемые артефакты рядом с входным медиа:

- `*.wav` нормализованное извлеченное аудио
- `*.srt` файл субтитров для плееров/редакторов
- `*.json` структурированные метаданные многоязычных субтитров

---

## 🛠 Использование

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Опции CLI

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### Поведение обработки

- Имена выходных файлов формируются от базового пути входа.
- Для `input.mp4` выходами будут `input.wav` (нормализованное аудио), `input.srt` (субтитры с таймкодами) и `input.json` (метаданные, включая `start`, `end`, `lang`, `text`, а также при наличии тайминги слов).
- Если `.srt` или `.json` уже существуют, запуск пропускается, если не указан `--force`.

---

## ⚙️ Конфигурация

Сейчас конфигурация в основном задается через CLI и значения по умолчанию в коде:

| Область конфигурации | Текущее поведение |
|---|---|
| Модель Whisper | `--whisper-model` (по умолчанию `large`) |
| Частота дискретизации обработки | Жестко задана `16000` для VAD/транскрибации |
| Извлечение FFmpeg | Mono WAV, `44100 Hz`, с `dynaudnorm=f=100` |
| Детектор Lingua | Инициализируется для `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` в основном потоке |
| Helper defaults для фильтрации на стороне Whisper | Включают `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Примечание о допущении: списки языков в helper defaults и в настройке основного детектора не полностью совпадают; этот README отражает текущее поведение реализации.

---

## 📦 Формат вывода

Инструмент пишет два артефакта субтитров на каждый входной медиафайл:

- `*.srt`: стандартные субтитры с таймкодами `HH:MM:SS,mmm`.
- `*.json`: структурированный список субтитров с форматированными таймкодами и языковыми метками.

Типичная форма JSON-сегмента:

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

Примечания:

- `start`/`end` сериализуются в JSON как строки в SRT-формате.
- `words` может присутствовать в зависимости от этапа обработки/уточнения сегмента.
- Значение `lang` может быть `und` для интервалов с неуверенным определением языка.

---

## 🧪 Примеры

Запуск на MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Запуск на MOV с принудительной перезаписью:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Запуск на аудио-входе, поддерживаемом FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Пример пакетного запуска в shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Заметки по разработке

- Канонический активный скрипт: `vad_lang_subtitle.py`.
- Исторические файлы (`*.old`, `*.shorterlength*`, `archived/`) полезны для справки, но выглядят неканоничными.
- В репозитории пока нет упаковочной обвязки проекта (`pyproject.toml`, `setup.py`) и нет закоммиченного CI/набора тестов.
- `data/` содержит крупные sample-медиа артефакты; учитывайте размер репозитория и расход локального диска в экспериментах.
- `clean_subtitles_dict()` присутствует в коде, но сейчас не вызывается основным пайплайном.
- `--force` — текущий механизм гарантированной перегенерации выходов при итеративной настройке.

Рекомендуемый локальный цикл разработки:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Используйте меньшую модель (`tiny`/`base`/`small`) во время итераций, затем переключайтесь на `large` для финального качества.

---

## 🩺 Устранение неполадок

| Симптом | Что делать |
|---|---|
| `ffmpeg: command not found` | Установите FFmpeg и проверьте через `ffmpeg -version`. |
| Первый запуск очень медленный или кажется зависшим | Первые загрузки моделей (Whisper + Silero) могут занять время; повторные запуски быстрее. |
| Ошибки CUDA / GPU | Попробуйте CPU fallback с меньшей моделью Whisper (`small`, `base`, `tiny`) и убедитесь, что сборка PyTorch соответствует вашему окружению. |
| Выходные файлы не пересоздаются | Используйте `--force`, чтобы перезаписать существующие производные файлы. |
| `pip install -r requirements.txt` падает из-за отсутствия файла | Используйте команду ручной установки зависимостей из раздела Installation. |
| Неточная языковая разметка на коротких сегментах | Такое бывает на очень коротких/шумных интервалах; текущая логика сочетает Whisper и Lingua, но пограничные случаи остаются. |
| Пустой или почти пустой файл субтитров | Убедитесь, что во входе есть речь, проверьте извлеченный `.wav` и повторите с `--force` после проверки извлечения FFmpeg. |
| Неожиданное переключение языка между соседними строками | Такое возможно на очень коротких сегментах; рассмотрите пост-объединение в downstream-инструментах по языку и минимальной длительности. |

---

## ⚠️ Известные ограничения и допущения

- Манифест зависимостей не закоммичен (`requirements.txt`, `pyproject.toml` и `setup.py` отсутствуют в корне репозитория на момент написания).
- Лицензия в README указана как MIT, но отдельный `LICENSE`-файл сейчас отсутствует.
- В основном потоке Lingua явно инициализируется с `EN/ZH/JA/AR`, тогда как helper defaults включает больше кандидатных кодов.
- Автоматические тесты/бенчмарки пока не закоммичены, поэтому валидация в основном ручная.
- В корне и в `archived/` есть исторические скрипты; активным следует считать только `vad_lang_subtitle.py`, если вы не проводите целенаправленные эксперименты.

---

## 🗺 Дорожная карта

- Добавить и поддерживать закрепленный `requirements.txt` или `pyproject.toml`.
- Добавить автоматические тесты для логики сегментации и очистки таймкодов.
- Добавить документацию по бенчмаркам и оценке качества для многоязычных пограничных случаев.
- Добавить поддержку опционального конфигурационного файла вместо поведения только на code-defaults.
- Расширить набор i18n README в `i18n/` и синхронизировать языковые панели.
- Уточнить и унифицировать поведение выбора языка между конфигурацией детектора и helper defaults.
- Добавить формальный файл `LICENSE`, соответствующий заявлению в README.

---

## ❤️ Support

Если проект экономит вам время, поддержка помогает финансировать сопровождение и дальнейшие улучшения.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

Дополнительные ссылки поддержки/сообщества:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) за speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) за надежное обнаружение голосовой активности
- [Lingua](https://github.com/pemistahl/lingua-java) за высокоточную идентификацию языка

---

## 🤝 Участие в проекте

1. Сделайте fork и clone.
2. Создайте ветку: `git checkout -b feat/your-idea`.
3. Сделайте commit и push.
4. Откройте PR.

Для существенных изменений приложите:

- Краткое описание ожидаемого изменения поведения
- Воспроизводимый пример команды
- Фрагменты субтитров до/после, где это уместно

---

## 📄 Лицензия

MIT © Lachlan Chen
