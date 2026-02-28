[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

مولّد ترجمات فرعية جاهز للاستخدام مبني على OpenAI Whisper، مع توسعة لاكتشاف اللغة بدقة على مستوى كل مقطع وتحسين النتائج للفيديوهات التي تحتوي على لغات مختلطة.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ نظرة عامة

`MultilingualWhisper` هو مسار عمل CLI بلغة Python يتمحور حول [`vad_lang_subtitle.py`](vad_lang_subtitle.py). وهو يجمع بين:

- Silero VAD لتجزئة الكلام
- OpenAI Whisper للتفريغ النصي والتنبؤ الأولي باللغة
- Lingua لتحسين تحديد اللغة اعتمادًا على النص
- FFmpeg للاستخراج والتطبيع ومعالجة الوسائط

المخرجات الأساسية هي ملفات ترجمة فرعية بصيغة `.srt` و`.json`، بالإضافة إلى صوت `.wav` مُستخرج ومُطبَّع.

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
  تنظيف الطوابع الزمنية يضمن عدم وجود فجوات أو تداخلات. تقسيم علامات الترقيم يفصل النصوص الطويلة عند الفواصل والنقاط وعلامات الاستفهام وغيرها. دمج VAD يعيد محاذاة الكلمات مع كتل VAD للحصول على ترجمات فرعية أكثر سلاسة. كما يُطبَّق تقسيم يعتمد على الطول بحدود خاصة بكل لغة.

- **ترجمات فرعية متعددة اللغات**  
  يُنتج كلًا من `.srt` و`.json` مع الحفاظ على وسوم اللغة لكل مقطع، بحيث يمكنك تنسيقها أو تصفيتها حسب اللغة في المشغلات أو أدوات التحرير اللاحقة.

- **معالجة وسائط قوية**  
  يستخرج الصوت ويطبّعه تلقائيًا عبر FFmpeg، ويحاول إصلاح الحاويات المعطوبة، ويطبّق التطبيع الديناميكي (`dynaudnorm`) للحصول على تفريغ أوضح.

---

## 🗂 هيكل المشروع

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

| العلم | الاسم المختصر | مطلوب | الوصف |
|---|---|---|---|
| `--video-path` | `-t` | نعم | مسار ملف الوسائط المدخل (فيديو/صوت مدعوم بواسطة FFmpeg) |
| `--whisper-model` | — | لا | اسم نموذج Whisper (الافتراضي: `large`) |
| `--force` | — | لا | إعادة التنفيذ حتى لو كانت ملفات `.wav` أو `.srt` أو `.json` موجودة مسبقًا |

### سلوك المعالجة

- أسماء المخرجات تُشتق من المسار الأساسي للمدخل.
- عند إدخال `input.mp4`، تكون المخرجات `input.wav` (صوت مُطبّع)، و`input.srt` (ترجمات زمنية)، و`input.json` (بيانات وصفية تشمل `start`, `end`, `lang`, `text`، وقد تتضمن توقيتات الكلمات اختياريًا).
- وجود `.srt` أو `.json` مسبقًا يؤدي إلى التخطي ما لم يتم تمرير `--force`.

---

## ⚙️ الإعدادات

الإعداد الحالي يعتمد أساسًا على خيارات CLI والقيم الافتراضية في الكود:

- نموذج Whisper: `--whisper-model` (الافتراضي `large`)
- معدل أخذ العينات: مضبوط في الكود على `16000` للمعالجة
- استخراج FFmpeg: WAV أحادي القناة، `44100 Hz`، مع `dynaudnorm=f=100`
- كاشف Lingua: مهيأ لـ `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` في التدفق الرئيسي
- رموز اللغة المسموح بها في تصفية جانب Whisper تتضمن `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` في القيم الافتراضية للدوال المساعدة

ملاحظة افتراضية: قوائم اللغات في القيم الافتراضية للدوال المساعدة وإعداد الكاشف الرئيسي ليست متطابقة بالكامل؛ هذا README يحافظ على السلوك الحالي كما هو مطبَّق.

---

## 🧪 أمثلة

تشغيل على ملف MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

تشغيل على MOV مع فرض الاستبدال:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

تشغيل على إدخال صوتي فقط ومدعوم بواسطة FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 ملاحظات التطوير

- السكربت النشط والمعتمد حاليًا هو `vad_lang_subtitle.py`.
- الملفات التاريخية (`*.old`, `*.shorterlength*`, `archived/`) مفيدة كمرجع لكنها تبدو غير معتمدة كأساس.
- لا توجد حاليًا بنية مشروع مُحزّمة (`pyproject.toml`, `setup.py`) ولا توجد مجموعة اختبارات/تكامل (CI) مُضافة.
- يحتوي `data/` على وسائط عيّنة كبيرة؛ انتبه إلى حجم المستودع واستهلاك مساحة القرص أثناء التجارب.
- الدالة `clean_subtitles_dict()` موجودة في الكود لكنها غير مستدعاه حاليًا في المسار الرئيسي.

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

---

## 🗺 خارطة الطريق

- إضافة وصيانة `requirements.txt` أو `pyproject.toml` مُثبت الإصدارات.
- إضافة اختبارات آلية لمنطق التجزئة وتنظيف الطوابع الزمنية.
- إضافة وثائق قياس الأداء وتقييم الجودة لحالات تعدد اللغات الطرفية.
- إضافة دعم ملف إعدادات اختياري بدل الاعتماد على القيم الافتراضية في الكود فقط.
- توسيع مجموعة README متعددة اللغات في `i18n/` والحفاظ على مزامنة شريط اللغات.

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

---

## 📄 الترخيص

MIT © Lachlan Chen
