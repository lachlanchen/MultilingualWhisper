[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

مولّد ترجمات فرعية جاهز للاستخدام مبني على OpenAI Whisper، مع توسعة لاكتشاف اللغة بدقة على مستوى كل مقطع وتحسين النتائج للفيديوهات التي تحتوي على لغات مختلطة.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## جدول المحتويات

- [نظرة عامة](#-نظرة-عامة)
- [لمحة سريعة](#لمحة-سريعة)
- [الميزات الرئيسية](#-الميزات-الرئيسية)
- [تدفق خط المعالجة](#-تدفق-خط-المعالجة)
- [هيكل المشروع](#-هيكل-المشروع)
- [المتطلبات المسبقة](#-المتطلبات-المسبقة)
- [التثبيت](#-التثبيت)
- [الاستخدام](#-الاستخدام)
- [الإعدادات](#-الإعدادات)
- [صيغة الإخراج](#-صيغة-الإخراج)
- [أمثلة](#-أمثلة)
- [ملاحظات التطوير](#-ملاحظات-التطوير)
- [استكشاف الأخطاء وإصلاحها](#-استكشاف-الأخطاء-وإصلاحها)
- [القيود والافتراضات المعروفة](#-القيود-والافتراضات-المعروفة)
- [خارطة الطريق](#-خارطة-الطريق)
- [الدعم](#-الدعم)
- [شكر وتقدير](#-شكر-وتقدير)
- [المساهمة](#-المساهمة)
- [الترخيص](#-الترخيص)

---

## ✨ نظرة عامة

`MultilingualWhisper` هو مسار CLI بلغة Python يتمحور حول [`vad_lang_subtitle.py`](vad_lang_subtitle.py). وهو يجمع بين:

- Silero VAD لتجزئة الكلام
- OpenAI Whisper للتفريغ النصي والتنبؤ الأولي باللغة
- Lingua لتحسين تحديد اللغة اعتمادًا على النص
- FFmpeg للاستخراج والتطبيع ومعالجة الوسائط

المخرجات الأساسية هي ملفات ترجمة فرعية بصيغة `.srt` و`.json`، بالإضافة إلى صوت `.wav` مُستخرج ومُطبّع.

### لمحة سريعة

| العنصر | التفاصيل |
|---|---|
| نقطة الدخول الرئيسية | `vad_lang_subtitle.py` |
| الإدخال | فيديو/صوت مدعوم بواسطة FFmpeg |
| الإخراج | `*.wav`, `*.srt`, `*.json` |
| التدفق الأساسي | VAD -> Whisper -> Lingua -> refinement |
| حالة الاستخدام المعتادة | إنشاء ترجمات فرعية متعددة اللغات |

---

## 🚀 الميزات الرئيسية

- **مسار Silero VAD -> Whisper**
  يقوم Voice Activity Detection (VAD) بتقسيم الصوت إلى مقاطع كلام، ثم يقوم Whisper بتفريغ كل جزء.

- **اكتشاف لغة دقيق على مستوى المقاطع**
  يستخدم [Lingua](https://github.com/pemistahl/lingua-java) إلى جانب كاشف Whisper الداخلي لوَسم كل مقطع (وحتى الكلمات المفردة) برموز لغة ISO مثل (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **تحسين ذكي للمقاطع**
  تنظيف الطوابع الزمنية يضمن عدم وجود فجوات أو تداخلات. تقسيم علامات الترقيم يفصل النصوص الطويلة عند الفواصل والنقاط وعلامات الاستفهام وغيرها. دمج VAD يعيد محاذاة الكلمات مع كتل VAD للحصول على ترجمات أكثر سلاسة. كما يُطبَّق تقسيم يعتمد على الطول بحدود خاصة بكل لغة.

- **ترجمات فرعية متعددة اللغات**
  يُنتج كلًا من `.srt` و`.json` مع الحفاظ على وسوم اللغة لكل مقطع، بحيث يمكنك تنسيقها أو تصفيتها حسب اللغة في المشغلات أو أدوات التحرير اللاحقة.

- **معالجة وسائط قوية**
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

المسار الرئيسي أثناء التشغيل في `vad_lang_subtitle.py`:

1. تحليل وسيطات CLI (`--video-path`, `--whisper-model`, `--force`).
2. تحديد مسارات الإخراج من الاسم الأساسي للمدخل.
3. استخراج/تطبيع الصوت عبر FFmpeg.
4. تحميل Silero VAD (`torch.hub`) ونموذج Whisper.
5. تنفيذ التفريغ النصي الأولي على مقاطع VAD.
6. دمج/تحسين المقاطع، ثم تنفيذ تفريغ ثانٍ على المقاطع المدمجة.
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

> ⚠️ ملاحظة: أشار README السابق إلى `requirements.txt`، لكنه غير موجود حاليًا في جذر المستودع.

---

## ✅ المتطلبات المسبقة

- Python `3.10+` (تم اختباره مع بيئات 3.x الحديثة)
- تثبيت `ffmpeg` وإتاحته على `PATH`
- موارد CPU/GPU + RAM كافية حسب نموذج Whisper المختار (لنموذج `large` يُوصى بشدة باستخدام GPU)
- اتصال إنترنت عند التشغيل الأول لجلب أوزان نماذج Whisper وأصول Silero VAD (عبر `torch.hub`)

حزم Python التي يستخدمها السكربت تشمل:

- `torch`
- `torchaudio`
- `whisper` (حزمة OpenAI Whisper للبايثون)
- `lingua-language-detector`
- `tqdm`

أوامر تحقق سريعة:

```bash
python --version
ffmpeg -version
```

---

## 🔧 التثبيت

1. **استنساخ هذا المستودع**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
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

إذا كان `requirements.txt` لا يزال غير موجود في نسختك المحلية، ثبّت اعتماديات التشغيل الأساسية يدويًا:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

وتأكد من تثبيت FFmpeg على مستوى النظام.

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
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### سلوك المعالجة

- أسماء المخرجات تُشتق من المسار الأساسي للمدخل.
- عند إدخال `input.mp4`، تكون المخرجات `input.wav` (صوت مُطبّع)، و`input.srt` (ترجمات زمنية)، و`input.json` (بيانات وصفية تشمل `start`, `end`, `lang`, `text`، وقد تتضمن توقيتات الكلمات اختياريًا).
- وجود `.srt` أو `.json` مسبقًا يؤدي إلى التخطي ما لم يتم تمرير `--force`.

---

## ⚙️ الإعدادات

الإعداد الحالي يعتمد أساسًا على خيارات CLI والقيم الافتراضية في الكود:

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

ملاحظة افتراضية: قوائم اللغات في القيم الافتراضية للدوال المساعدة وإعداد الكاشف الرئيسي ليست متطابقة بالكامل؛ هذا README يحافظ على السلوك الحالي كما هو مطبَّق.

---

## 📦 صيغة الإخراج

تكتب الأداة ملفي ترجمة فرعية لكل ملف وسائط مدخل:

- `*.srt`: نص ترجمة فرعية قياسي بطوابع زمنية `HH:MM:SS,mmm`.
- `*.json`: قائمة ترجمة منظمة تحتوي على طوابع زمنية منسقة ووسوم اللغة.

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

- `start`/`end` يتم تسلسلهما كسلاسل بنمط SRT ضمن إخراج JSON.
- الحقل `words` قد يظهر اعتمادًا على مرحلة معالجة/تحسين المقاطع.
- قد تظهر قيمة `lang` تساوي `und` للمقاطع ذات اللغة غير المؤكدة.

---

## 🧪 أمثلة

تشغيل على ملف MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

تشغيل على MOV مع فرض إعادة الكتابة:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

تشغيل على إدخال صوتي فقط ومدعوم بواسطة FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

مثال دفعي عبر shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 ملاحظات التطوير

- السكربت النشط والمعتمد حاليًا هو `vad_lang_subtitle.py`.
- الملفات التاريخية (`*.old`, `*.shorterlength*`, `archived/`) مفيدة كمرجع لكنها تبدو غير معتمدة كأساس.
- لا توجد حاليًا بنية مشروع مُحزّمة (`pyproject.toml`, `setup.py`) ولا توجد مجموعة اختبارات/تكامل (CI) مُضافة.
- يحتوي `data/` على وسائط عيّنة كبيرة؛ انتبه إلى حجم المستودع واستهلاك مساحة القرص أثناء التجارب.
- الدالة `clean_subtitles_dict()` موجودة في الكود لكنها غير مستدعاه حاليًا في المسار الرئيسي.
- `--force` هي الآلية الحالية لضمان إعادة توليد المخرجات أثناء الضبط التكراري.

حلقة تطوير محلية مقترحة:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

استخدم نموذجًا أصغر (`tiny`/`base`/`small`) أثناء التكرار، ثم انتقل إلى `large` للحصول على جودة نهائية أعلى.

---

## 🩺 استكشاف الأخطاء وإصلاحها

| العَرَض | ما الذي ينبغي فعله |
|---|---|
| `ffmpeg: command not found` | ثبّت FFmpeg وتحقق باستخدام `ffmpeg -version`. |
| التشغيل الأول بطيء جدًا أو يبدو متوقفًا | تنزيلات النماذج الأولية (Whisper + Silero) قد تستغرق وقتًا؛ عمليات التشغيل اللاحقة أسرع. |
| أخطاء CUDA / GPU | جرّب الرجوع إلى CPU باستخدام نموذج Whisper أصغر (`small`, `base`, `tiny`) وتأكد من توافق نسخة PyTorch مع بيئتك. |
| ملفات الإخراج لا تُعاد كتابتها | استخدم `--force` لاستبدال الملفات المشتقة الموجودة. |
| فشل `pip install -r requirements.txt` بسبب عدم وجود الملف | استخدم أمر التثبيت اليدوي للاعتماديات الموضح في قسم التثبيت. |
| دقة وسم اللغة منخفضة في المقاطع القصيرة | قد يحدث ذلك في المقاطع القصيرة جدًا أو المليئة بالضوضاء؛ المنطق الحالي يجمع بين Whisper وLingua لكنه ما زال يواجه حالات طرفية. |
| إخراج ترجمة فرعية فارغ أو شبه فارغ | تأكد من أن الإدخال يحتوي على كلام، وافحص ملف `.wav` المستخرج، ثم أعد المحاولة مع `--force` بعد التحقق من استخراج FFmpeg. |
| تقلب غير متوقع للغة بين الأسطر المتجاورة | قد يحدث هذا في المقاطع القصيرة جدًا؛ فكر في الدمج اللاحق في أدوات المصب حسب اللغة والحد الأدنى للمدة. |

---

## ⚠️ القيود والافتراضات المعروفة

- لم يتم اعتماد ملف توصيف للاعتماديات في المستودع حتى الآن (`requirements.txt` و`pyproject.toml` و`setup.py` غير موجودة في الجذر وقت كتابة هذا النص).
- الترخيص مذكور في README على أنه MIT، لكن ملف `LICENSE` مستقل غير موجود حاليًا.
- يتم تهيئة Lingua صراحةً مع `EN/ZH/JA/AR` في المسار الرئيسي، بينما تتضمن القيم الافتراضية المساعدة رموزًا إضافية.
- لا توجد اختبارات/معايير أداء آلية مُعتمدة حاليًا، لذا فإن التحقق يتم يدويًا بشكل أساسي.
- توجد سكربتات تاريخية في الجذر و`archived/`؛ يجب اعتبار `vad_lang_subtitle.py` هو الملف النشط ما لم يكن هناك قصد واضح للتجريب.

---

## 🗺 خارطة الطريق

- إضافة وصيانة `requirements.txt` أو `pyproject.toml` مُثبت الإصدارات.
- إضافة اختبارات آلية لمنطق التجزئة وتنظيف الطوابع الزمنية.
- إضافة وثائق قياس الأداء وتقييم الجودة لحالات تعدد اللغات الطرفية.
- إضافة دعم ملف إعدادات اختياري بدل الاعتماد على القيم الافتراضية في الكود فقط.
- توسيع مجموعة README متعددة اللغات في `i18n/` والحفاظ على مزامنة شريط اللغات.
- توضيح وتوحيد سلوك اختيار اللغة بين إعدادات الكاشف والقيم الافتراضية للدوال المساعدة.
- إضافة ملف `LICENSE` رسمي ليتوافق مع التصريح الموجود في README.

---

## 💖 الدعم

إذا كان هذا المشروع مفيدًا لك، يمكنك دعم التطوير عبر:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- الموقع الشخصي: https://lazying.art
- الدردشة/المجتمع: https://chat.lazying.art
- مركز الأفكار/المشاريع: https://onlyideas.art

---

## 🔗 شكر وتقدير

- [OpenAI Whisper](https://github.com/openai/whisper) لتقنية تحويل الكلام إلى نص
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) لاكتشاف النشاط الصوتي بكفاءة عالية
- [Lingua](https://github.com/pemistahl/lingua-java) لتحديد اللغات بدقة عالية

---

## 🤝 المساهمة

1. قم بعمل Fork ثم Clone
2. أنشئ فرعًا: `git checkout -b feat/your-idea`
3. نفّذ Commit وPush
4. افتح PR

للتغييرات الكبيرة، يرجى تضمين:

- وصفًا موجزًا للتغيير المتوقع في السلوك
- مثال أمر قابل لإعادة الإنتاج
- مقتطفات ترجمة فرعية قبل/بعد عند الحاجة

---

## 📄 الترخيص

MIT © Lachlan Chen
