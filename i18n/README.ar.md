[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

مولِّد ترجمات فرعية جاهز للاستخدام مبني على OpenAI Whisper، ومُوسَّع باكتشاف دقيق للغة على مستوى كل مقطع مع تحسين خاص للفيديوهات التي تحتوي على لغات مختلطة.

> أنشئ ترجمات فرعية متعددة اللغات أنظف من وسائط حقيقية مختلطة لغويًا عبر تقسيم واعٍ باللغة.

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

> 🌍 **توثيق متعدد اللغات متاح**: الإنجليزية + 10 نسخ README مترجمة داخل [`i18n/`](i18n/)، وروابطها في شريط اللغات أعلاه.

### لغات التوثيق

| اللغة | الملف |
| --- | --- |

| المحور | القيمة |
| --- | --- |
| الإدخال | صوت/فيديو متوافق مع FFmpeg |
| خط المعالجة | تقسيم VAD -> تفريغ Whisper -> تنقيح Lingua |
| المخرجات | `*.wav` و`*.srt` و`*.json` بعد التطبيع |
| أفضل استخدام | ترجمات متعددة اللغات مع وسم لغة لكل مقطع |

---

## جدول المحتويات

- [نظرة عامة](#-نظرة-عامة)
- [لمحة سريعة](#لمحة-سريعة)
- [الميزات الرئيسية](#-الميزات-الرئيسية)
- [تدفق خط المعالجة](#-تدفق-خط-المعالجة)
- [هيكل المشروع](#-هيكل-المشروع)
- [المتطلبات المسبقة](#-المتطلبات-المسبقة)
- [التثبيت](#-التثبيت)
- [البدء السريع](#-البدء-السريع)
- [دليل اختيار النموذج](#-دليل-اختيار-النموذج)
- [الاستخدام](#-الاستخدام)
- [الإعدادات](#-الإعدادات)
- [تنسيق الإخراج](#-تنسيق-الإخراج)
- [أمثلة](#-أمثلة)
- [ملاحظات التطوير](#-ملاحظات-التطوير)
- [استكشاف الأخطاء وإصلاحها](#-استكشاف-الأخطاء-وإصلاحها)
- [القيود والافتراضات المعروفة](#-القيود-والافتراضات-المعروفة)
- [خطة الطريق](#-خطة-الطريق)
- [الشكر والتقدير](#-الشكر-والتقدير)
- [المساهمة](#-المساهمة)
- [الدعم](#-support)
- [التواصل](#-التواصل)
- [الترخيص](#-الترخيص)

---

## ✨ نظرة عامة

`MultilingualWhisper` هو خط أنابيب CLI بلغة Python يتمحور حول [`vad_lang_subtitle.py`](vad_lang_subtitle.py). ويجمع بين:

- Silero VAD لتقسيم الكلام
- OpenAI Whisper للتفريغ النصي والتنبؤ الأولي باللغة
- Lingua لتنقيح اللغة اعتمادًا على النص
- FFmpeg للاستخراج والتطبيع ومعالجة الوسائط

المخرجات الأساسية هي ملفات ترجمة فرعية بصيغة `.srt` و`.json`، بالإضافة إلى ملف صوتي `.wav` مستخرج ومطبع.

### لمحة سريعة

| العنصر | التفاصيل |
|---|---|
| نقطة الدخول الرئيسية | `vad_lang_subtitle.py` |
| الإدخال | فيديو/صوت مدعوم من FFmpeg |
| المخرجات | `*.wav`, `*.srt`, `*.json` |
| التدفق الأساسي | VAD -> Whisper -> Lingua -> التنقيح |
| حالة الاستخدام الشائعة | توليد ترجمات متعددة اللغات |

---

## 🚀 الميزات الرئيسية

- **خط معالجة Silero VAD -> Whisper**  
  يقوم Voice Activity Detection (VAD) بتقسيم الصوت إلى مقاطع كلام، ثم يقوم Whisper بتفريغ كل جزء.

- **اكتشاف لغة دقيق على مستوى المقاطع**  
  يستخدم [Lingua](https://github.com/pemistahl/lingua-java) إلى جانب كاشف Whisper نفسه لوَسم كل مقطع (حتى الكلمات المفردة) برموز لغة ISO مثل (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **تنقيح ذكي للمقاطع**  
  تنظيف الطوابع الزمنية يضمن عدم وجود فجوات أو تداخلات. تقسيم الترقيم يجزّئ التفريغ الطويل عند الفواصل والنقاط وعلامات الاستفهام وغيرها. دمج VAD يعيد محاذاة الكلمات إلى كتل VAD للحصول على ترجمات أكثر سلاسة. كما يطبّق التقسيم المعتمد على الطول حدودًا خاصة بكل لغة.

- **ترجمات فرعية متعددة اللغات**  
  يخرج `.srt` و`.json` مع الحفاظ على وسوم اللغة لكل مقطع، بحيث يمكنك التنسيق أو التصفية حسب اللغة في المشغلات أو المحررات اللاحقة.

- **معالجة وسائط متينة**  
  يستخرج الصوت ويطبّعه تلقائيًا عبر FFmpeg، ويحاول إصلاح الحاويات المعطوبة، ويطبق التطبيع الديناميكي (`dynaudnorm`) للحصول على تفريغ أوضح.

---

## 🔁 تدفق خط المعالجة

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

المسار التشغيلي الرئيسي داخل `vad_lang_subtitle.py`:

1. تحليل وسيطات CLI (`--video-path`, `--whisper-model`, `--force`).
2. تحديد مسارات الإخراج من اسم الملف الأساسي للإدخال.
3. استخراج/تطبيع الصوت عبر FFmpeg.
4. تحميل Silero VAD (`torch.hub`) ونموذج Whisper.
5. تفريغ مبدئي على مقاطع VAD.
6. دمج/تنقيح المقاطع، ثم تفريغ ثانٍ على النطاقات المدمجة.
7. تطبيق تقليل طول الترجمة وتنظيف الطوابع الزمنية.
8. حفظ `.srt` و`.json`.

---

## 🗂 هيكل المشروع

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

> ⚠️ ملاحظة: كانت النسخ السابقة من README تشير إلى `requirements.txt`، لكنه مفقود حاليًا من جذر المستودع.

---

## ✅ المتطلبات المسبقة

- Python `3.10+` (مُختبر على بيئات 3.x الحديثة)
- تثبيت `ffmpeg` وإتاحته عبر `PATH`
- موارد CPU/GPU + RAM كافية لنموذج Whisper المختار (لنموذج `large` يُنصح بقوة باستخدام GPU)
- اتصال إنترنت في أول تشغيل لجلب أوزان Whisper وأصول Silero VAD (`torch.hub`)

حزم Python المستخدمة في السكربت تشمل:

- `torch`
- `torchaudio`
- `whisper` (حزمة OpenAI Whisper لبايثون)
- `lingua-language-detector`
- `tqdm`

أوامر تحقق سريعة:

```bash
python --version
ffmpeg -version
```

---

## 🔧 التثبيت

1. **استنسخ المستودع**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **أنشئ بيئة افتراضية وفعّلها**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **ثبّت الاعتماديات**

```bash
pip install -r requirements.txt
```

إذا كان `requirements.txt` لا يزال غير موجود في نسختك، ثبّت اعتماديات التشغيل الأساسية يدويًا:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

وتأكد من تثبيت FFmpeg على مستوى النظام.

---

## ⚡ البدء السريع

إذا أردت أسرع طريق من الاستنساخ إلى الترجمة الفرعية:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

نصيحة: استخدم `small` أثناء التكرار، ثم انتقل إلى `large` عند التصدير النهائي عالي الجودة.

الملفات المتوقعة بجانب وسيط الإدخال:

- `*.wav` صوت مستخرج ومطبع
- `*.srt` ملف ترجمة فرعية للمشغلات/المحررات
- `*.json` بيانات وصفية متعددة اللغات منظّمة

---

## 🎚 دليل اختيار النموذج

اختر نموذج Whisper بناءً على مفاضلة السرعة مقابل الجودة:

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | Fastest | Lowest | Fast smoke tests and pipeline validation |
| `small` | Fast | Good | Daily iteration and local development |
| `medium` | Medium | Better | Balanced production workflows |
| `large` (default) | Slowest | Best | Final subtitle exports for highest quality |

نمط عملي:

1. كرّر باستخدام `small --force`
2. تحقّق من التوقيت ووسوم اللغة
3. أعد التشغيل باستخدام `large --force` لإخراج التسليم

---

## 🛠 الاستخدام

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### خيارات CLI

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | مسار الوسائط المدخلة (فيديو/صوت يدعمه FFmpeg) |
| `--whisper-model` | — | No | اسم نموذج Whisper (الافتراضي: `large`) |
| `--force` | — | No | إعادة التشغيل حتى لو كانت `.wav` أو `.srt` أو `.json` موجودة مسبقًا |

### سلوك المعالجة

- يتم اشتقاق أسماء الإخراج من المسار الأساسي للإدخال.
- بالنسبة إلى `input.mp4`، تكون المخرجات: `input.wav` (صوت مطبع)، و`input.srt` (ترجمة بطوابع زمنية)، و`input.json` (بيانات وصفية تشمل `start`, `end`, `lang`, `text`، وأحيانًا توقيت الكلمات).
- وجود `.srt` أو `.json` مسبقًا يؤدي إلى التخطي إلا إذا تم تمرير `--force`.

---

## ⚙️ الإعدادات

الإعداد الحالي مدفوع بشكل أساسي عبر CLI والقيم الافتراضية في الكود:

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

ملاحظة افتراضية: قوائم اللغات في القيم الافتراضية للمساعد وإعداد الكاشف الرئيسي ليست متطابقة بالكامل؛ هذا README يحافظ على السلوك الحالي كما هو مطبق.

تفصيل إضافي من السكربت الحالي:

- يتم تطبيق `torch.set_num_threads(1)` أثناء التشغيل.
- يتم تحميل نموذج VAD من `snakers4/silero-vad` عبر `torch.hub.load(...)`.
- تنظيف المقاطع يحذف الإدخالات التي لغتها `und` أو نصها فارغ.

---

## 📦 تنسيق الإخراج

تنتج الأداة ملفي ترجمة فرعية لكل وسيط إدخال:

- `*.srt`: نص ترجمة قياسي بطوابع زمنية `HH:MM:SS,mmm`.
- `*.json`: قائمة ترجمة منظّمة تحتوي على طوابع زمنية منسقة ووسوم لغة.

شكل مقطع JSON نموذجي:

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

ملاحظات:

- يتم تسلسل `start`/`end` كسلاسل بنمط SRT داخل JSON.
- قد يكون `words` موجودًا حسب مرحلة المعالجة/التنقيح للمقطع.
- قد تظهر قيمة `lang` كـ `und` للمقاطع غير المؤكدة لغويًا.

---

## 🧪 أمثلة

التشغيل على ملف MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

التشغيل على MOV مع فرض الكتابة فوق الملفات:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

التشغيل على إدخال صوتي فقط يدعمه FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

مثال دفعي (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 ملاحظات التطوير

- السكربت النشط المعتمد هو `vad_lang_subtitle.py`.
- الملفات التاريخية (`*.old`, `*.shorterlength*`, `archived/`) مفيدة كمرجع لكنها تبدو غير أساسية.
- لا توجد حاليًا هيكلة مشروع مُحزّمة (`pyproject.toml`, `setup.py`) ولا مجموعة CI/اختبارات ملتزمة.
- يحتوي `data/` على عينات وسائط كبيرة؛ انتبه لحجم المستودع واستهلاك التخزين المحلي أثناء التجارب.
- الدالة `clean_subtitles_dict()` موجودة في الكود لكنها غير مستخدمة حاليًا في المسار الرئيسي.
- `--force` هي الآلية الحالية لضمان إعادة توليد المخرجات أثناء الضبط التكراري.

حلقة تطوير محلية مقترحة:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

استخدم نموذجًا أصغر (`tiny`/`base`/`small`) أثناء التكرار، ثم انتقل إلى `large` للحصول على أفضل جودة نهائية.

---

## 🩺 استكشاف الأخطاء وإصلاحها

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | ثبّت FFmpeg وتحقق عبر `ffmpeg -version`. |
| First run is very slow or appears stuck | تنزيل النماذج أول مرة (Whisper + Silero) قد يستغرق وقتًا؛ التشغيلات اللاحقة أسرع. |
| CUDA / GPU errors | جرّب CPU باستخدام نموذج Whisper أصغر (`small`, `base`, `tiny`) وتأكد من توافق نسخة PyTorch مع بيئتك. |
| Output files are not regenerated | استخدم `--force` للكتابة فوق الملفات المشتقة الموجودة. |
| `pip install -r requirements.txt` fails because file not found | استخدم أمر تثبيت الاعتماديات اليدوي المذكور في قسم Installation. |
| Inaccurate language tagging on short segments | قد يحدث هذا في المقاطع القصيرة جدًا/المليئة بالضجيج؛ المنطق الحالي يدمج Whisper وLingua لكنه لا يزال يواجه حالات طرفية. |
| Empty or near-empty subtitle output | تأكد أن الإدخال يحتوي كلامًا، وافحص ملف `.wav` المستخرج، ثم أعد التشغيل باستخدام `--force` بعد التحقق من استخراج FFmpeg. |
| Unexpected language flips between neighboring lines | قد يحدث هذا مع المقاطع القصيرة جدًا؛ فكر في الدمج لاحقًا في أدوات downstream حسب اللغة والمدة الدنيا. |
| FFmpeg extraction fails on damaged media | يحاول السكربت إعادة المحاولة بعد إصلاح الحاوية (`-c copy -movflags +faststart`)، لكن الملفات شديدة التلف قد تفشل. |

تشخيصات سريعة:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ القيود والافتراضات المعروفة

- بيان الاعتماديات غير ملتزم (`requirements.txt`, `pyproject.toml`, و`setup.py` غائبة من الجذر وقت كتابة هذا النص).
- الترخيص معلن كـ MIT في README، لكن ملف `LICENSE` مستقل غير موجود حاليًا.
- يتم تهيئة Lingua صراحةً بـ `EN/ZH/JA/AR` في المسار الرئيسي، بينما تتضمن القيم الافتراضية للمساعد رموزًا مرشحة إضافية.
- لا توجد اختبارات/معايير أداء آلية ملتزمة حاليًا، لذا التحقق يتم يدويًا بشكل أساسي.
- توجد سكربتات تاريخية في الجذر و`archived/`؛ يجب اعتبار `vad_lang_subtitle.py` هو النشط ما لم تكن هناك تجارب مقصودة.
- السكربت يطبع حاليًا سجلات تشغيل مطولة ومخرجات تصحيح لكل مقطع؛ وهذا سلوك متوقع في التنفيذ الحالي.

---

## 🗺 خطة الطريق

- إضافة وصيانة `requirements.txt` مثبت أو `pyproject.toml`.
- إضافة اختبارات آلية لمنطق التقسيم وتنظيف الطوابع الزمنية.
- إضافة وثائق قياس أداء وجودة لحالات الحافة متعددة اللغات.
- إضافة دعم ملف إعداد اختياري بدل السلوك المعتمد فقط على القيم الافتراضية في الكود.
- توسيع مجموعة README متعددة اللغات في `i18n/` مع إبقاء أشرطة اللغات متزامنة.
- توضيح وتوحيد سلوك اختيار اللغة بين إعدادات الكاشف والقيم الافتراضية للمساعد.
- إضافة ملف `LICENSE` رسمي ليتطابق مع تصريح README.

---

## 🔗 الشكر والتقدير

- [OpenAI Whisper](https://github.com/openai/whisper) لتحويل الكلام إلى نص
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) لاكتشاف نشاط الصوت بشكل متين
- [Lingua](https://github.com/pemistahl/lingua-java) لتحديد اللغة بدقة عالية

---

## 🤝 المساهمة

1. Fork وclone
2. أنشئ فرعًا: `git checkout -b feat/your-idea`
3. نفّذ commit وpush
4. افتح PR

بالنسبة للتغييرات الكبيرة، أرفق:

- وصفًا قصيرًا للتغيير السلوكي المتوقع
- مثال أمر قابل لإعادة الإنتاج
- مقتطفات ترجمة قبل/بعد عند الحاجة

---

## 📫 التواصل

- افتح issue للإبلاغ عن الأخطاء، وأسئلة الاستخدام، وطلبات الميزات.
- استخدم خيارات الدعم أعلاه لاستفسارات الرعاية والتبرع.

---

## 📄 الترخيص

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
