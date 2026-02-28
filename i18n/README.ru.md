[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Генератор субтитров формата "подключил и используешь", построенный на OpenAI Whisper и расширенный точным определением и уточнением языка для каждого сегмента в видео со смешанной речью на разных языках.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents

- [Обзор](#-обзор)
- [Кратко](#кратко)
- [Ключевые возможности](#-ключевые-возможности)
- [Поток конвейера](#-поток-конвейера)
- [Структура проекта](#-структура-проекта)
- [Требования](#-требования)
- [Установка](#-установка)
- [Использование](#-использование)
- [Конфигурация](#️-конфигурация)
- [Формат вывода](#-формат-вывода)
- [Примеры](#-примеры)
- [Заметки для разработки](#-заметки-для-разработки)
- [Устранение неполадок](#-устранение-неполадок)
- [Известные ограничения и допущения](#️-известные-ограничения-и-допущения)
- [Дорожная карта](#-дорожная-карта)
- [Поддержка](#-поддержка)
- [Благодарности](#-благодарности)
- [Участие в проекте](#-участие-в-проекте)
- [Лицензия](#-лицензия)

---

## ✨ Обзор

`MultilingualWhisper` — это Python CLI-конвейер, построенный вокруг [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Он сочетает:

- Silero VAD для сегментации речи
- OpenAI Whisper для транскрибации и первичного определения языка
- Lingua для уточнения языка на основе текста
- FFmpeg для извлечения, нормализации и обработки медиа

Основные выходные артефакты: файлы субтитров `.srt` и `.json`, а также извлеченное нормализованное аудио `.wav`.

### Кратко

| Пункт | Детали |
|---|---|
| Основная точка входа | `vad_lang_subtitle.py` |
| Вход | Видео/аудио, поддерживаемые FFmpeg |
| Выход | `*.wav`, `*.srt`, `*.json` |
| Основной поток | VAD -> Whisper -> Lingua -> refinement |
| Типичный сценарий | Генерация субтитров для смешанной речи на разных языках |

---

## 🚀 Ключевые возможности

- **Конвейер Silero VAD -> Whisper**
  Voice Activity Detection (VAD) делит аудио на речевые сегменты, после чего Whisper транскрибирует каждый фрагмент.

- **Детальное определение языка**
  Использует [Lingua](https://github.com/pemistahl/lingua-java) вместе со встроенным детектором Whisper, чтобы помечать каждый сегмент (и даже отдельные слова) ISO-кодами языков (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Интеллектуальное уточнение сегментов**
  Очистка таймкодов гарантирует отсутствие разрывов и наложений. Разбиение по пунктуации делит длинные транскрипции по запятым, точкам, вопросительным знакам и т.д. Объединение по VAD повторно выравнивает слова обратно к VAD-блокам, чтобы субтитры читались плавнее. Сегментация с учетом длины применяет языко-специфичные лимиты.

- **Мультиязычные субтитры**
  Генерирует и `.srt`, и `.json`, сохраняя языковые метки для каждого сегмента, чтобы вы могли стилизовать или фильтровать субтитры по языку в последующих плеерах и редакторах.

- **Надежная обработка медиа**
  Автоматически извлекает и нормализует аудио через FFmpeg, пытается восстанавливать поврежденные контейнеры и применяет динамическую нормализацию (`dynaudnorm`) для более чистой транскрибации.

---

## 🔁 Поток конвейера

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

Основной путь выполнения в `vad_lang_subtitle.py`:

1. Разбор аргументов CLI (`--video-path`, `--whisper-model`, `--force`).
2. Вычисление путей выходных файлов на основе базового имени входного файла.
3. Извлечение/нормализация аудио через FFmpeg.
4. Загрузка Silero VAD (`torch.hub`) и модели Whisper.
5. Первый проход транскрибации по VAD-фрагментам.
6. Объединение/уточнение сегментов, затем второй проход транскрибации по объединенным интервалам.
7. Применение уменьшения длины субтитров и очистки таймкодов.
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

> ⚠️ Примечание: в предыдущем README упоминался `requirements.txt`, но сейчас этот файл отсутствует в корне репозитория.

---

## ✅ Требования

- Python `3.10+` (проверено в современных окружениях 3.x)
- Установленный `ffmpeg`, доступный в `PATH`
- Достаточный CPU/GPU и объем RAM для выбранной модели Whisper (для `large` настоятельно рекомендуется GPU)
- Доступ в интернет при первом запуске для загрузки весов Whisper и ассетов Silero VAD (`torch.hub`)

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

Если в вашем рабочем дереве по-прежнему нет `requirements.txt`, установите основные runtime-зависимости вручную:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

И убедитесь, что FFmpeg установлен на уровне системы.

---

## 🛠 Использование

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Параметры CLI

| Флаг | Алиас | Обязательный | Описание |
|---|---|---|---|
| `--video-path` | `-t` | Да | Путь к входному медиа (видео/аудио, поддерживаемые FFmpeg) |
| `--whisper-model` | — | Нет | Название модели Whisper (по умолчанию: `large`) |
| `--force` | — | Нет | Повторный запуск, даже если `.wav`, `.srt` или `.json` уже существуют |

### Поведение при обработке

- Имена выходных файлов формируются из базового пути входного файла.
- Для `input.mp4` выходами будут `input.wav` (нормализованное аудио), `input.srt` (субтитры с таймкодами) и `input.json` (метаданные, включая `start`, `end`, `lang`, `text`, а также при необходимости тайминги слов).
- Если `.srt` или `.json` уже существуют, обработка будет пропущена, если не установлен `--force`.

---

## ⚙️ Конфигурация

Сейчас конфигурация в основном задается через CLI и значения по умолчанию в коде:

| Область конфигурации | Текущее поведение |
|---|---|
| Модель Whisper | `--whisper-model` (по умолчанию `large`) |
| Частота дискретизации для обработки | Жестко задана как `16000` для VAD/транскрибации |
| Извлечение через FFmpeg | Mono WAV, `44100 Hz`, с `dynaudnorm=f=100` |
| Детектор Lingua | Инициализируется для `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` в основном потоке |
| Значения по умолчанию в helper для фильтрации со стороны Whisper | Включают `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Примечание о допущении: списки языков в helper-defaults и в настройке основного детектора не полностью совпадают; этот README сохраняет текущее поведение в том виде, как оно реализовано.

---

## 📦 Формат вывода

Инструмент создает два артефакта субтитров для каждого входного медиа:

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

- `start`/`end` сериализуются в JSON как строки в стиле SRT.
- Поле `words` может присутствовать в зависимости от этапа обработки/уточнения сегмента.
- Значение `lang` может быть `und` для интервалов с неопределенным языком.

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

Пример batch-обработки в shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Заметки для разработки

- Канонический активный скрипт — `vad_lang_subtitle.py`.
- Исторические файлы (`*.old`, `*.shorterlength*`, `archived/`) полезны как справка, но выглядят неканоничными.
- В репозитории пока нет упакованного каркаса проекта (`pyproject.toml`, `setup.py`) и закоммиченного CI/набора тестов.
- `data/` содержит крупные артефакты sample-медиа; учитывайте размер репозитория и расход локального диска в ходе экспериментов.
- `clean_subtitles_dict()` присутствует в коде, но сейчас не вызывается основным конвейером.
- `--force` — текущий механизм, который гарантирует перегенерацию выходов при итеративной настройке.

Рекомендуемый локальный цикл разработки:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Используйте меньшую модель (`tiny`/`base`/`small`) во время итераций, а затем переключайтесь на `large` для финального качества вывода.

---

## 🩺 Устранение неполадок

| Симптом | Что делать |
|---|---|
| `ffmpeg: command not found` | Установите FFmpeg и проверьте через `ffmpeg -version`. |
| Первый запуск очень медленный или выглядит зависшим | Первичная загрузка моделей (Whisper + Silero) может занять время; повторные запуски быстрее. |
| Ошибки CUDA / GPU | Попробуйте fallback на CPU, используя меньшую модель Whisper (`small`, `base`, `tiny`), и убедитесь, что сборка PyTorch соответствует вашему окружению. |
| Выходные файлы не пересоздаются | Используйте `--force` для перезаписи существующих производных файлов. |
| `pip install -r requirements.txt` завершается ошибкой, потому что файл не найден | Используйте команду ручной установки зависимостей из раздела Installation. |
| Неточная языковая разметка на коротких сегментах | Такое возможно на очень коротких/шумных интервалах; текущая логика сочетает Whisper и Lingua, но пограничные случаи остаются. |
| Пустой или почти пустой файл субтитров на выходе | Проверьте, что во входе есть речь, посмотрите извлеченный `.wav` и повторите запуск с `--force` после проверки извлечения через FFmpeg. |
| Неожиданные переключения языка между соседними строками | Это может происходить на очень коротких сегментах; рассмотрите пост-объединение в downstream-инструментах по языку и минимальной длительности. |

---

## ⚠️ Известные ограничения и допущения

- Манифест зависимостей не закоммичен (`requirements.txt`, `pyproject.toml` и `setup.py` отсутствуют в корне репозитория на момент написания).
- Лицензия указана в README как MIT, но отдельный файл `LICENSE` сейчас отсутствует.
- Lingua в основном потоке явно инициализируется с `EN/ZH/JA/AR`, тогда как defaults в helper включают больше кандидатных кодов.
- Автоматические тесты/бенчмарки пока не закоммичены, поэтому валидация в основном ручная.
- В корне и в `archived/` присутствуют исторические скрипты; активным следует считать только `vad_lang_subtitle.py`, если вы не проводите целенаправленные эксперименты.

---

## 🗺 Дорожная карта

- Добавить и поддерживать зафиксированный `requirements.txt` или `pyproject.toml`.
- Добавить автоматические тесты для логики сегментации и очистки таймкодов.
- Добавить документацию по бенчмаркам и оценке качества для мультиязычных пограничных случаев.
- Добавить поддержку опционального конфигурационного файла вместо поведения только на code-defaults.
- Расширить набор i18n README в `i18n/` и синхронизировать языковые панели.
- Уточнить и унифицировать поведение выбора языка между конфигурацией детектора и defaults в helper.
- Добавить формальный файл `LICENSE`, соответствующий заявлению в README.

---

## 💖 Поддержка

Если этот проект вам помогает, вы можете поддержать разработку через:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) за speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) за надежное определение голосовой активности
- [Lingua](https://github.com/pemistahl/lingua-java) за высокоточную идентификацию языка

---

## 🤝 Участие в проекте

1. Сделайте fork и clone
2. Создайте ветку: `git checkout -b feat/your-idea`
3. Сделайте commit и push
4. Откройте PR

Для существенных изменений добавьте:

- Короткое описание ожидаемого изменения поведения
- Воспроизводимый пример команды
- Фрагменты субтитров до/после, где это уместно

---

## 📄 Лицензия

MIT © Lachlan Chen
