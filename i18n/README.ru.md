[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Генератор субтитров «подставь и работай», построенный на OpenAI Whisper и расширенный точным определением и уточнением языка для каждого сегмента в видео со смешанными языками.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Обзор

`MultilingualWhisper` — это Python CLI-конвейер, сосредоточенный вокруг [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Он объединяет:

- Silero VAD для сегментации речи
- OpenAI Whisper для транскрибации и начального определения языка
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
| Типичный сценарий | Генерация субтитров для смешанных языков |

---

## 🚀 Ключевые возможности

- **Конвейер Silero VAD -> Whisper**  
  Voice Activity Detection (VAD) разбивает аудио на речевые сегменты, после чего Whisper транскрибирует каждый фрагмент.

- **Точное определение языка**  
  Использует [Lingua](https://github.com/pemistahl/lingua-java) вместе со встроенным детектором Whisper, чтобы помечать каждый сегмент (и даже отдельные слова) ISO-кодами языков (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Интеллектуальное уточнение сегментов**  
  Очистка таймкодов убирает разрывы и пересечения. Разбиение по пунктуации делит длинные транскрипции по запятым, точкам, вопросительным знакам и т.д. Повторное выравнивание по VAD возвращает слова в VAD-блоки для более плавных субтитров. Сегментация с учетом длины применяет языко-специфичные лимиты.

- **Мультиязычные субтитры**  
  Выводит и `.srt`, и `.json`, сохраняя языковые метки для каждого сегмента, чтобы вы могли стилизовать или фильтровать субтитры по языку в последующих плеерах или редакторах.

- **Надежная обработка медиа**  
  Автоматически извлекает и нормализует аудио через FFmpeg, пытается исправлять поврежденные контейнеры и применяет динамическую нормализацию (`dynaudnorm`) для более четкой транскрибации.

---

## 🗂 Структура проекта

```text
.
├── README.md
├── vad_lang_subtitle.py               # Основной конвейер: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old           # Устаревший прототип
├── vad_lang_subtitle.py.20250706      # Исторический снимок
├── vad_lang_subtitle.py.shorterlength # Альтернативный исторический вариант
├── vad_lang_subtitle.py.shorterlength2# Альтернативный исторический вариант
├── vad_lang_subtitle.srt              # Пример выходного файла
├── vad_lang_subtitle.json             # Пример JSON
├── .github/
│   └── FUNDING.yml                    # Ссылки на поддержку
├── archived/                          # Старые эксперименты/прототипы
├── data/                              # Опциональные примеры медиа + сгенерированные выходы
├── figs/                              # Брендинговые ассеты (баннер/логотип)
├── i18n/                              # Рабочая область переводов/readme (сейчас присутствует, пустая)
└── .auto-readme-work/                 # Артефакты рабочей области генерации README
```

> ⚠️ Примечание: в предыдущем README упоминался `requirements.txt`, но сейчас он отсутствует в корне репозитория.

---

## ✅ Требования

- Python `3.10+` (проверено в современных окружениях 3.x)
- Установленный `ffmpeg`, доступный в `PATH`
- Достаточно CPU/GPU и RAM для выбранной модели Whisper (для `large` настоятельно рекомендуется GPU)
- Доступ в интернет при первом запуске для загрузки весов Whisper и ассетов Silero VAD (`torch.hub`)

Python-пакеты, используемые скриптом:

- `torch`
- `torchaudio`
- `whisper` (Python-пакет OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

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

| Флаг | Алиас | Обязателен | Описание |
|---|---|---|---|
| `--video-path` | `-t` | Да | Путь к входному медиа (видео/аудио, поддерживаемые FFmpeg) |
| `--whisper-model` | — | Нет | Имя модели Whisper (по умолчанию: `large`) |
| `--force` | — | Нет | Повторный запуск, даже если `.wav`, `.srt` или `.json` уже существуют |

### Поведение обработки

- Имена выходных файлов формируются из базового пути входа.
- Для `input.mp4` выходами будут `input.wav` (нормализованное аудио), `input.srt` (субтитры с таймкодами) и `input.json` (метаданные с `start`, `end`, `lang`, `text`, опционально таймингами слов).
- Если `.srt` или `.json` уже существуют, обработка пропускается, если не задан `--force`.

---

## ⚙️ Конфигурация

Сейчас конфигурация в основном задается через CLI и значения по умолчанию в коде:

- Модель Whisper: `--whisper-model` (по умолчанию `large`)
- Частота дискретизации: жестко задана как `16000` для обработки
- Извлечение через FFmpeg: mono WAV, `44100 Hz`, с `dynaudnorm=f=100`
- Детектор Lingua: инициализируется для `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` в основном потоке
- Разрешенные языковые коды для фильтрации на стороне Whisper в helper-defaults включают `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`

Примечание по допущениям: языковые списки в helper-defaults и в настройке основного детектора не полностью совпадают; этот README сохраняет текущее поведение в реализованном виде.

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

---

## 🧭 Заметки для разработки

- Канонический активный скрипт — `vad_lang_subtitle.py`.
- Исторические файлы (`*.old`, `*.shorterlength*`, `archived/`) полезны для справки, но выглядят неканоничными.
- Сейчас в репозитории нет упаковочной структуры проекта (`pyproject.toml`, `setup.py`) и нет зафиксированного CI/набора тестов.
- `data/` содержит крупные sample-медиа-артефакты; учитывайте размер репозитория и использование локального диска во время экспериментов.
- `clean_subtitles_dict()` присутствует в коде, но сейчас не вызывается основным конвейером.

---

## 🩺 Устранение неполадок

| Симптом | Что делать |
|---|---|
| `ffmpeg: command not found` | Установите FFmpeg и проверьте через `ffmpeg -version`. |
| Первый запуск очень медленный или будто завис | Первичная загрузка моделей (Whisper + Silero) может занять время; повторные запуски быстрее. |
| Ошибки CUDA / GPU | Попробуйте fallback на CPU, выбрав меньшую модель Whisper (`small`, `base`, `tiny`), и убедитесь, что сборка PyTorch соответствует вашему окружению. |
| Выходные файлы не пересоздаются | Используйте `--force` для перезаписи существующих производных файлов. |
| `pip install -r requirements.txt` падает из-за отсутствия файла | Используйте команду ручной установки зависимостей из раздела Installation. |
| Неточная языковая разметка на коротких сегментах | Такое возможно на очень коротких/шумных отрезках; текущая логика сочетает Whisper и Lingua, но пограничные случаи остаются. |

---

## 🗺 Дорожная карта

- Добавить и поддерживать зафиксированный `requirements.txt` или `pyproject.toml`.
- Добавить автоматические тесты для логики сегментации и очистки таймкодов.
- Добавить документацию по бенчмаркам и оценке качества для мультиязычных пограничных случаев.
- Добавить поддержку опционального файла конфигурации вместо поведения только с code-defaults.
- Расширить набор i18n README в `i18n/` и синхронизировать языковые панели.

---

## 💖 Поддержка

Если проект вам полезен, вы можете поддержать разработку:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Personal site: https://lazying.art
- Chat/community: https://chat.lazying.art
- Ideas/project hub: https://onlyideas.art

---

## 🔗 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) за speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) за надежное определение голосовой активности
- [Lingua](https://github.com/pemistahl/lingua-java) за высокоточное определение языка

---

## 🤝 Участие в проекте

1. Сделайте fork и clone
2. Создайте ветку: `git checkout -b feat/your-idea`
3. Сделайте commit и push
4. Откройте PR

---

## 📄 Лицензия

MIT © Lachlan Chen
