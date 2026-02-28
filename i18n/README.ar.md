[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

مولّد ترجمات فرعية جاهز للاستخدام مبني على OpenAI Whisper، مع دعم دقيق لاكتشاف اللغة على مستوى كل مقطع وتحسين النتائج لفيديوهات تحتوي على لغات مختلطة.

> أنشئ ترجمات فرعية متعددة اللغات أنظف من وسائط الواقع الحقيقي المختلطة لغويًا عبر تقسيم واعٍ باللغة.

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

| التركيز | القيمة |
| --- | --- |
| الإدخال | ملف صوت/فيديو متوافق مع FFmpeg |
| خط المعالجة | تقسيم VAD → تفريغ Whisper → تنقيح Lingua |
| النتيجة | ملفات `*.wav` و`*.srt` و`*.json` بعد التطبيع |
| أفضل استخدام | ترجمات متعددة اللغات مع وسوم لغة لكل مقطع |

---

## جدول المحتويات

- [نظرة عامة](#-نظرة-عامة)
- [لمحة سريعة](#-لمحة-سريعة)
- [الميزات الرئيسية](#-الميزات-الرئيسية)
- [تدفق خط المعالجة](#-تدفق-خط-المعالجة)
- [هيكل المشروع](#-هيكل-المشروع)
- [المتطلبات المسبقة](#-المتطلبات-المسبقة)
- [التثبيت](#-التثبيت)
- [البدء السريع](#-البدء-السريع)
- [الاستخدام](#-الاستخدام)
- [الإعدادات](#-الإعدادات)
- [تنسيق الإخراج](#-تنسيق-الإخراج)
- [الأمثلة](#-الأمثلة)
- [ملاحظات التطوير](#-ملاحظات-التطوير)
- [استكشاف الأخطاء وإصلاحها](#-استكشاف-الأخطاء-وإصلاحها)
- [القيود والافتراضات المعروفة](#-القيود-والافتراضات-المعروفة)
- [خطة الطريق](#-خطة-الطريق)
- [الدعم](#-support)
- [الاتصال](#-الاتصال)
- [الشكر والتقدير](#-الشكر-والتقدير)
- [المساهمة](#-المساهمة)
- [الترخيص](#-الترخيص)

---

## ✨ نظرة عامة

`MultilingualWhisper` هو خط أنابيب CLI بلغة Python مركزي حول [`vad_lang_subtitle.py`](vad_lang_subtitle.py). يجمع بين:

- Silero VAD لتجزئة الكلام
- OpenAI Whisper للتفريغ النصي والتنبؤ الأولي باللغة
- Lingua لتنقيح اللغة بالنص
- FFmpeg للاستخراج والتطبيع ومعالجة الوسائط

الإخراجات الأساسية هي ملفات الترجمة الفرعية بصيغة `.srt` و`.json`، بالإضافة إلى ملف صوتي `.wav` مستخرج ومُطَبَّع.

### لمحة سريعة

| العنصر | التفاصيل |
|---|---|
| نقطة الدخول الرئيسية | `vad_lang_subtitle.py` |
| الإدخال | فيديو/صوت مدعوم بواسطة FFmpeg |
| الإخراج | `*.wav`, `*.srt`, `*.json` |
| التدفق الأساسي | VAD -> Whisper -> Lingua -> التنقيح |
| حالة الاستخدام الشائعة | إنشاء ترجمات متعددة اللغات |

---

## 🚀 الميزات الرئيسية

- **خط VAD -> Whisper**
  يقوم كشف نشاط الصوت (VAD) بتقسيم الصوت إلى مقاطع كلامية، ثم يقوم Whisper بتفريغ كل مقطع.

- **اكتشاف لغة دقيق على مستوى المقاطع**
  يستخدم [Lingua](https://github.com/pemistahl/lingua-java) إلى جانب كاشف Whisper الأصلي لوسم كل مقطع (حتى الكلمات المفردة) برموز ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **تنقيح ذكي للمقاطع**
  تنظيف الطوابع الزمنية يضمن عدم وجود فجوات أو تداخلات. تقسيم الجمل بعلامات الترقيم يقسم التفريغ الطويل عند الفواصل والنقاط وعلامات الاستفهام، وغيرها. تُمَزّج مقاطع VAD لإعادة محاذاة الكلمات مع كتل VAD لتصبح الترجمة الفرعية أكثر سلاسة. ويُطبّق تقسيم واعٍ للطول يحدّد حدودًا مخصصة حسب اللغة.

- **ترجمات فرعية متعددة اللغات**
  ينتج ملفات `.srt` و`.json` مع الحفاظ على وسوم اللغة لكل مقطع حتى تتمكن من تعديل النمط أو التصفية حسب اللغة في المشغلات أو أدوات التحرير اللاحقة.

- **معالجة وسائط قوية**
  يستخرج الصوت تلقائيًا ويُطبّع عبر FFmpeg، ويحاول إصلاح الحاويات التالفة، ويطبق التطبيع الديناميكي (`dynaudnorm`) للحصول على تفريغ أوضح.

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

المسار الرئيسي أثناء التشغيل في `vad_lang_subtitle.py`:

1. تحليل وسيطات CLI (`--video-path`, `--whisper-model`, `--force`).
2. استخراج مسارات الإخراج من اسم الأساس للملف المدخل.
3. استخراج/تطبيع الصوت عبر FFmpeg.
4. تحميل Silero VAD (`torch.hub`) ونموذج Whisper.
5. تفريغ أولي على مقاطع VAD.
6. دمج/تنقيح المقاطع، ثم تفريغ ثانٍ على النطاقات المدمجة.
7. تطبيق تقليل طول الترجمة وتنظيف الطوابع الزمنية.
8. حفظ ملفات `.srt` و`.json`.

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

> ⚠️ ملاحظة: كانت الإصدارات السابقة من README تشير إلى `requirements.txt`، لكنه غير موجود حاليًا في جذر المستودع.

---

## ✅ المتطلبات المسبقة

- Python `3.10+` (تم اختباره على بيئات 3.x الحديثة)
- `ffmpeg` مثبت ومتاح في `PATH`
- موارد CPU/GPU + RAM كافية لنموذج Whisper المختار (لنموذج `large` يُنصح بشدة باستخدام GPU)
- اتصال إنترنت في أول تشغيل لجلب أوزان نماذج Whisper ومكونات Silero VAD (`torch.hub`)

تتضمن حزم Python التي يستخدمها السكربت:

- `torch`
- `torchaudio`
- `whisper` (حزمة OpenAI Whisper لبايثون)
- `lingua-language-detector`
- `tqdm`

أوامر التحقق السريعة:

```bash
python --version
ffmpeg -version
```

---

## 🔧 التثبيت

1. **استنساخ هذا المستودع**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **إنشاء وتفعيل بيئة افتراضية**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **تثبيت الاعتماديات**

```bash
pip install -r requirements.txt
```

إذا كان ملف `requirements.txt` غير موجود في نسخة العمل الحالية، ثبّت الاعتماديات الأساسية يدويًا:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

وتأكد من أن FFmpeg مثبّت على مستوى النظام.

---

## ⚡ البدء السريع

إذا أردت أسرع طريقة من الاستنساخ إلى الترجمة الفرعية:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

نصيحة: استخدم `small` أثناء التجريب، ثم بدّل إلى `large` عند الإخراج النهائي.

الملفات المتوقعة بجانب الوسيط المدخل:

- `*.wav` صوت مستخرج ومطبع
- `*.srt` ملف ترجمة فرعية للمشغلات/المحررات
- `*.json` بيانات وصفية متعددة اللغات منظمة

---

## 🛠 الاستخدام

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### خيارات CLI

| المعلمة | البديل | مطلوب | الوصف |
|---|---|---|---|
| `--video-path` | `-t` | نعم | مسار الوسائط المدخل (فيديو/صوت مدعوم بواسطة FFmpeg) |
| `--whisper-model` | — | لا | اسم نموذج Whisper (افتراضي: `large`) |
| `--force` | — | لا | أعد التنفيذ حتى لو كانت `.wav` أو `.srt` أو `.json` موجودة بالفعل |

### سلوك المعالجة

- تُشتق أسماء الإخراج من المسار الأساسي للمدخل.
- بالنسبة لملف `input.mp4`، تكون المخرجات هي `input.wav` (صوت مطبّع)، `input.srt` (ترجمة مرقمة زمنيا)، و`input.json` (بيانات وصفية تتضمن `start`, `end`, `lang`, `text`، وتوقيتات الكلمات اختياريًا).
- وجود ملف `.srt` أو `.json` مسبقًا يؤدي إلى التخطي ما لم يتم تعيين `--force`.

---

## ⚙️ الإعدادات

الإعداد الحالي يعتمد أساسًا على وسيطات CLI والقيم الافتراضية المعرفة في الكود:

| مجال الإعداد | السلوك الحالي |
|---|---|
| نموذج Whisper | `--whisper-model` (افتراضي: `large`) |
| معدل العينة للمعالجة | مضبوط مسبقًا على `16000` لمعالجة VAD/التفريغ النصي |
| استخراج FFmpeg | WAV أحادي القناة، `44100 Hz`، مع `dynaudnorm=f=100` |
| كاشف Lingua | يُهيَّأ لـ `ENGLISH` و`CHINESE` و`JAPANESE` و`ARABIC` في المسار الرئيسي |
| القيم الافتراضية لمصفاة Whisper الجانبية | تشمل `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

ملاحظة افتراضية: قوائم اللغات في القيم الافتراضية للمصفاة وإعداد الكاشف الرئيسي ليست متطابقة بالكامل؛ هذا الـREADME يعكس السلوك الحالي كما هو مطبق.

---

## 📦 تنسيق الإخراج

تكتب الأداة ملفي ترجمة فرعية لكل وسيط مدخل:

- `*.srt`: نص ترجمة فرعية قياسي مع طوابع زمنية `HH:MM:SS,mmm`.
- `*.json`: قائمة ترجمة مهيكلة تحتوي على طوابع زمنية منسقة ووسوم اللغة.

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

- `start`/`end` تُسلسَل كسلاسل بنمط SRT داخل إخراج JSON.
- قد يظهر الحقل `words` حسب مرحلة معالجة/تنقيح المقطع.
- قد يظهر قيمة `lang` كـ `und` لمقاطع لغة غير مؤكدة.

---

## 🧪 أمثلة

التنفيذ على MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

التنفيذ على MOV مع الإعادة القسرية:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

التنفيذ على وسيط صوتي فقط يدعمه FFmpeg:

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

- السكربت النشط والمرجعي هو `vad_lang_subtitle.py`.
- الملفات التاريخية (`*.old`, `*.shorterlength*`, `archived/`) مفيدة كمرجع لكنها غير معيارية.
- لا يوجد حاليًا بنية مشروع مكتبة (`pyproject.toml`, `setup.py`) ولا توجد CI أو اختبارات مرفوعة.
- يحتوي المجلد `data/` على عناصر وسائط كبيرة؛ انتبه لحجم المستودع واستخدام القرص المحلي أثناء التجارب.
- دالة `clean_subtitles_dict()` موجودة في الكود لكنها غير مستخدمة حاليًا في خط المعالجة الرئيسي.
- `--force` هو الآلية الحالية لضمان إعادة إنشاء المخرجات أثناء التعديل التدريجي.

اقتراح حلقة تطوير محلية:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

استخدم نموذجًا أصغر (`tiny`/`base`/`small`) أثناء التكرار، ثم انتقل إلى `large` لتحسين الجودة النهائية.

---

## 🩺 استكشاف الأخطاء وإصلاحها

| العرض | الإجراء |
|---|---|
| `ffmpeg: command not found` | ثبّت FFmpeg وتحقق عبر `ffmpeg -version`. |
| أول تشغيل بطيء جدًا أو يبدو متوقفًا | تنزيلات النماذج الأولية (Whisper + Silero) قد تستغرق وقتًا؛ إعادة التشغيل تكون أسرع. |
| أخطاء CUDA / GPU | جرّب الرجوع إلى CPU باستخدام نموذج Whisper أصغر (`small`, `base`, `tiny`) وتأكد من توافق نسخة PyTorch مع البيئة. |
| ملفات الإخراج لا تُعاد إنشاؤها | استخدم `--force` لكتابة الملفات المشتقة الموجودة من جديد. |
| فشل `pip install -r requirements.txt` لعدم وجود الملف | استخدم أمر تثبيت الاعتماديات اليدوي الموجود في قسم التثبيت. |
| وسم لغة غير دقيق في المقاطع القصيرة | قد يحدث ذلك في المقاطع القصيرة جدًا أو المليئة بالضوضاء؛ المنطق الحالي يجمع بين Whisper وLingua لكنه ما زال يواجه حالات طرفية. |
| إخراج ترجمة فرعية فارغ أو شبه فارغ | تأكد أن الإدخال يحتوي على كلام، وافحص ملف `.wav` المستخرج، ثم أعد المحاولة باستخدام `--force` بعد التحقق من استخراج FFmpeg. |
| تقلبات لغة غير متوقعة بين السطور المتجاورة | قد يحدث هذا مع المقاطع القصيرة جدًا؛ فكّر في الدمج اللاحق حسب اللغة والمدة الأدنى في أدوات المصب اللاحقة. |

---

## ⚠️ القيود والافتراضات المعروفة

- لا يوجد ملف بيان التبعية في المستودع (`requirements.txt`, `pyproject.toml`, و`setup.py` غير موجودة في الجذر الآن).
- يذكر الترخيص MIT في README، لكن ملف `LICENSE` المنفصل غير موجود حاليًا.
- يتم تهيئة Lingua صراحة بـ `EN/ZH/JA/AR` في المسار الرئيسي، بينما تتضمن القيم الافتراضية المساعدة رموزًا مرشحة إضافية.
- لا توجد اختبارات/معايير أداء تلقائية مضافة حاليًا، لذلك تتم المراجعة يدويًا غالبًا.
- توجد سكربتات تاريخية في الجذر و`archived/`؛ يجب اعتبار `vad_lang_subtitle.py` فقط هو الملف النشط ما لم يكون هناك تجارب مقصودة.

---

## 🗺 خطة الطريق

- إضافة وصيانة `requirements.txt` أو `pyproject.toml` مثبت الإصدار.
- إضافة اختبارات تلقائية لمنطق تقسيم المقاطع وتنظيف الطوابع الزمنية.
- إضافة وثائق قياس وأداء وتقييم الجودة لحالات الحواف متعددة اللغات.
- إضافة دعم لملف إعدادات اختياري بدل السلوك المعتمد فقط على القيم الافتراضية في الكود.
- توسيع ملفات README متعددة اللغات في `i18n/` والحفاظ على توحيد شريط اللغات.
- توضيح وتوحيد سلوك اختيار اللغة بين إعدادات الكاشف والقيم الافتراضية للمساعد.
- إضافة ملف `LICENSE` رسمي يتماشى مع ما هو مذكور في README.

---

## 🔗 الشكر والتقدير

- [OpenAI Whisper](https://github.com/openai/whisper) للتفريغ النصي لكلام إلى نص
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) لاكتشاف نشاط الصوت بقوة
- [Lingua](https://github.com/pemistahl/lingua-java) لتحديد اللغة بدقة عالية

---

## 📫 الاتصال

- افتح issue للإبلاغ عن الأخطاء، وطرح أسئلة الاستخدام، وطلب الميزات.
- استخدم خيارات الدعم أعلاه لطلبات الرعاية أو التبرع.

---

## 📄 الترخيص

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
