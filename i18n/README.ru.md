[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Генератор субтитров «подставь и работай» на базе OpenAI Whisper, расширенный точным определением языка для каждого сегмента и дополнительной обработкой видео со смешанной речью на разных языках.

> Создавайте более чистые многоязычные субтитры из реальных медиа со смешанными языками благодаря языково-осведомленной сегментации.

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

> 🌍 **Доступна многоязычная документация**: английская версия + 10 переводов README в каталоге [`i18n/`](../i18n/), ссылки также есть в языковой панели выше.

### Языки документации

| Язык | Файл |
| --- | --- |

| Фокус | Значение |
| --- | --- |
| Вход | Аудио/видео, совместимые с FFmpeg |
| Пайплайн | VAD-сегментация -> Whisper-транскрибация -> Lingua-уточнение |
| Выход | Нормализованный `*.wav`, `*.srt` и `*.json` |
| Лучший сценарий | Субтитры для смешанной многоязычной речи с тегами языка для каждого сегмента |

---

## Содержание

- [Обзор](#-обзор)
- [Кратко](#кратко)
- [Ключевые возможности](#-ключевые-возможности)
- [Схема пайплайна](#-схема-пайплайна)
- [Структура проекта](#-структура-проекта)
- [Требования](#-требования)
- [Установка](#-установка)
- [Быстрый старт](#-быстрый-старт)
- [Руководство по выбору модели](#-руководство-по-выбору-модели)
- [Использование](#-использование)
- [Конфигурация](#-конфигурация)
- [Формат выходных данных](#-формат-выходных-данных)
- [Примеры](#-примеры)
- [Примечания по разработке](#-примечания-по-разработке)
- [Устранение неполадок](#-устранение-неполадок)
- [Известные ограничения и допущения](#-известные-ограничения-и-допущения)
- [План развития](#-план-развития)
- [Благодарности](#-благодарности)
- [Как внести вклад](#-как-внести-вклад)
- [Support](#-support)
- [Контакты](#-контакты)
- [Лицензия](#-лицензия)

---

## ✨ Обзор

`MultilingualWhisper` — это CLI-пайплайн на Python, в центре которого находится [`vad_lang_subtitle.py`](../vad_lang_subtitle.py). Он объединяет:

- Silero VAD для сегментации речи
- OpenAI Whisper для транскрибации и начального определения языка
- Lingua для уточнения языка по тексту
- FFmpeg для извлечения, нормализации и обработки медиа

Основные результаты — файлы субтитров `.srt` и `.json`, а также извлеченное нормализованное аудио `.wav`.

### Кратко

| Пункт | Детали |
|---|---|
| Основная точка входа | `vad_lang_subtitle.py` |
| Вход | Видео/аудио, поддерживаемые FFmpeg |
| Выход | `*.wav`, `*.srt`, `*.json` |
| Основной поток | VAD -> Whisper -> Lingua -> refinement |
| Типовой сценарий | Генерация субтитров для смешанной многоязычной речи |

---

## 🚀 Ключевые возможности

- **Пайплайн Silero VAD -> Whisper**  
  Voice Activity Detection (VAD) разбивает аудио на речевые сегменты, после чего Whisper транскрибирует каждый фрагмент.

- **Точное определение языка**  
  Использует [Lingua](https://github.com/pemistahl/lingua-java) вместе со встроенным детектором Whisper, чтобы помечать каждый сегмент (и даже отдельные слова) ISO-кодами языка (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Интеллектуальная доработка сегментов**  
  Очистка таймкодов устраняет разрывы и пересечения. Разбиение по пунктуации делит длинные транскрипции по запятым, точкам, вопросительным знакам и т.д. Слияние на базе VAD повторно выравнивает слова по VAD-блокам для более плавных субтитров. Сегментация с учетом длины применяет языко-специфичные ограничения.

- **Многоязычные субтитры**  
  Формирует и `.srt`, и `.json`, сохраняя языковые теги у каждого сегмента, чтобы можно было стилизовать или фильтровать субтитры по языку в последующих плеерах/редакторах.

- **Надежная обработка медиа**  
  Автоматически извлекает и нормализует аудио через FFmpeg, пытается восстановить поврежденные контейнеры и применяет динамическую нормализацию (`dynaudnorm`) для более чистой транскрибации.

---

## 🔁 Схема пайплайна

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
2. Определение выходных путей из базового имени входного файла.
3. Извлечение/нормализация аудио через FFmpeg.
4. Загрузка Silero VAD (`torch.hub`) и модели Whisper.
5. Первая транскрибация по VAD-фрагментам.
6. Слияние/доработка сегментов, затем вторая транскрибация по объединенным интервалам.
7. Применение сокращения длины субтитров и очистка таймкодов.
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

> ⚠️ Примечание: в предыдущей версии README упоминался `requirements.txt`, но сейчас он отсутствует в корне репозитория.

---

## ✅ Требования

- Python `3.10+` (проверено в современных окружениях Python 3.x)
- установленный `ffmpeg`, доступный в `PATH`
- достаточные CPU/GPU и RAM для выбранной модели Whisper (для `large` настоятельно рекомендуется GPU)
- интернет при первом запуске для загрузки весов Whisper и ресурсов Silero VAD (`torch.hub`)

Скрипт использует следующие Python-пакеты:

- `torch`
- `torchaudio`
- `whisper` (Python-пакет OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

Команды для быстрой проверки:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Установка

1. **Клонируйте репозиторий**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
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

Если в вашей копии по-прежнему нет `requirements.txt`, установите основные зависимости вручную:

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

Совет: используйте `small` во время итераций, затем переключитесь на `large` для финального качества.

Ожидаемые артефакты рядом с входным медиа:

- `*.wav` извлеченное нормализованное аудио
- `*.srt` файл субтитров для плееров/редакторов
- `*.json` структурированные метаданные многоязычных субтитров

---

## 🎚 Руководство по выбору модели

Выбирайте модель Whisper исходя из баланса скорости и качества:

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | Fastest | Lowest | Fast smoke tests and pipeline validation |
| `small` | Fast | Good | Daily iteration and local development |
| `medium` | Medium | Better | Balanced production workflows |
| `large` (default) | Slowest | Best | Final subtitle exports for highest quality |

Практический шаблон:

1. Итерируйтесь с `small --force`
2. Проверьте тайминги и языковые теги
3. Перезапустите с `large --force` для финального результата

---

## 🛠 Использование

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Параметры CLI

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### Поведение обработки

- Имена выходных файлов формируются из базового пути входного файла.
- Для `input.mp4` выходами будут `input.wav` (нормализованное аудио), `input.srt` (субтитры с таймкодами) и `input.json` (метаданные, включая `start`, `end`, `lang`, `text`, опционально тайминги слов).
- При наличии существующих `.srt` или `.json` обработка пропускается, если не установлен `--force`.

---

## ⚙️ Конфигурация

Текущая конфигурация в основном задается через CLI и значения по умолчанию в коде:

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Примечание по допущениям: списки языков в значениях по умолчанию у вспомогательных функций и в инициализации основного детектора не полностью совпадают; этот README сохраняет текущее поведение реализации.

Дополнительные детали реализации из текущего скрипта:

- Во время выполнения применяется `torch.set_num_threads(1)`.
- Модель VAD загружается из `snakers4/silero-vad` через `torch.hub.load(...)`.
- Очистка сегментов удаляет записи, где язык `und` или текст пуст.

---

## 📦 Формат выходных данных

Инструмент записывает два артефакта субтитров на каждый входной медиафайл:

- `*.srt`: стандартный текст субтитров с таймкодами `HH:MM:SS,mmm`.
- `*.json`: структурированный список субтитров с отформатированными таймкодами и тегами языка.

Типичная структура сегмента в JSON:

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

- `start`/`end` сериализуются как строки в SRT-формате в JSON.
- `words` могут присутствовать в зависимости от этапа обработки/уточнения сегмента.
- Значение `lang` равное `und` может появляться для интервалов с неопределенным языком.

---

## 🧪 Примеры

Запуск для MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Запуск для MOV с принудительной перезаписью:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Запуск для входного аудио, поддерживаемого FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Пример пакетной обработки в shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Примечания по разработке

- Канонический активный скрипт: `vad_lang_subtitle.py`.
- Исторические файлы (`*.old`, `*.shorterlength*`, `archived/`) полезны для справки, но выглядят неканоничными.
- Сейчас в репозитории нет упаковочной структуры проекта (`pyproject.toml`, `setup.py`) и нет закоммиченного CI/набора тестов.
- `data/` содержит крупные тестовые медиаартефакты; учитывайте размер репозитория и потребление диска при экспериментах.
- `clean_subtitles_dict()` существует в коде, но в основном пайплайне сейчас не вызывается.
- `--force` — текущий механизм гарантированной перегенерации выходов при итеративной настройке.

Рекомендуемый локальный цикл разработки:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Используйте меньшую модель (`tiny`/`base`/`small`) во время итераций, затем переключайтесь на `large` для финального качества.

---

## 🩺 Устранение неполадок

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

Быстрая диагностика:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Известные ограничения и допущения

- Файл зависимостей не закоммичен (`requirements.txt`, `pyproject.toml` и `setup.py` отсутствуют в корне репозитория на момент написания).
- В README указана лицензия MIT, но отдельный файл `LICENSE` сейчас отсутствует.
- В основном потоке Lingua явно инициализируется с `EN/ZH/JA/AR`, тогда как значения по умолчанию у вспомогательных функций содержат больше кодов.
- Автоматические тесты/бенчмарки пока не закоммичены, поэтому валидация в основном ручная.
- В корне и в `archived/` присутствуют исторические скрипты; активным следует считать только `vad_lang_subtitle.py`, если вы не проводите целевые эксперименты.
- Скрипт сейчас выводит подробные runtime-логи и отладочный вывод по сегментам; это ожидаемое поведение текущей реализации.

---

## 🗺 План развития

- Добавить и поддерживать закрепленный `requirements.txt` или `pyproject.toml`.
- Добавить автоматические тесты для логики сегментации и очистки таймкодов.
- Добавить документацию по бенчмаркам и оценке качества для сложных многоязычных случаев.
- Добавить поддержку опционального конфигурационного файла вместо поведения только на кодовых значениях по умолчанию.
- Расширить набор i18n README в `i18n/` и поддерживать синхронизацию языковых панелей.
- Уточнить и унифицировать выбор языков между конфигурацией детектора и значениями по умолчанию вспомогательных функций.
- Добавить формальный файл `LICENSE` в соответствии с декларацией в README.

---

## 🔗 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) за speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) за надежное определение речевой активности
- [Lingua](https://github.com/pemistahl/lingua-java) за высокоточную идентификацию языка

---

## 🤝 Как внести вклад

1. Сделайте fork и clone
2. Создайте ветку: `git checkout -b feat/your-idea`
3. Сделайте commit и push
4. Откройте PR

Для крупных изменений добавьте:

- Краткое описание ожидаемого изменения поведения
- Воспроизводимый пример команды
- Фрагменты субтитров «до/после», если уместно

---

## 📫 Контакты

- Открывайте issue для сообщений об ошибках, вопросов по использованию и запросов на новые функции.
- Используйте варианты поддержки выше по вопросам спонсорства и пожертвований.

---

## 📄 Лицензия

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
