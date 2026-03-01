[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Trình tạo phụ đề dùng ngay, xây trên OpenAI Whisper, được mở rộng với khả năng phát hiện và tinh chỉnh ngôn ngữ chính xác theo từng đoạn cho video chứa nhiều ngôn ngữ.

> Tạo phụ đề đa ngôn ngữ sạch hơn từ media thực tế có ngôn ngữ pha trộn, với phân đoạn nhận biết ngôn ngữ.

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

> 🌍 **Tài liệu đa ngôn ngữ đã có**: English + 10 biến thể README đã dịch trong [`i18n/`](i18n/), được liên kết ở thanh ngôn ngữ phía trên.

### Ngôn ngữ tài liệu

| Locale | File |
| --- | --- |

| Trọng tâm | Giá trị |
| --- | --- |
| Input | Âm thanh/video tương thích FFmpeg |
| Pipeline | VAD segmentation -> Whisper transcription -> Lingua refinement |
| Output | `*.wav`, `*.srt`, và `*.json` đã chuẩn hóa |
| Trường hợp phù hợp nhất | Phụ đề đa ngôn ngữ với nhãn ngôn ngữ theo từng đoạn |

---

## Table of Contents

- [Tổng quan](#-tổng-quan)
- [Nhìn nhanh](#nhìn-nhanh)
- [Tính năng chính](#-tính-năng-chính)
- [Luồng pipeline](#-luồng-pipeline)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Điều kiện tiên quyết](#-điều-kiện-tiên-quyết)
- [Cài đặt](#-cài-đặt)
- [Bắt đầu nhanh](#-bắt-đầu-nhanh)
- [Hướng dẫn chọn model](#-hướng-dẫn-chọn-model)
- [Cách dùng](#-cách-dùng)
- [Cấu hình](#-cấu-hình)
- [Định dạng đầu ra](#-định-dạng-đầu-ra)
- [Ví dụ](#-ví-dụ)
- [Ghi chú phát triển](#-ghi-chú-phát-triển)
- [Khắc phục sự cố](#-khắc-phục-sự-cố)
- [Hạn chế và giả định đã biết](#-hạn-chế-và-giả-định-đã-biết)
- [Lộ trình](#-lộ-trình)
- [Lời cảm ơn](#-lời-cảm-ơn)
- [Đóng góp](#-đóng-góp)
- [Support](#-support)
- [Liên hệ](#-liên-hệ)
- [Giấy phép](#-giấy-phép)

---

## ✨ Tổng quan

`MultilingualWhisper` là một pipeline CLI Python xoay quanh [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Nó kết hợp:

- Silero VAD để phân đoạn tiếng nói
- OpenAI Whisper để phiên âm và dự đoán ngôn ngữ ban đầu
- Lingua để tinh chỉnh ngôn ngữ dựa trên văn bản
- FFmpeg để trích xuất, chuẩn hóa và xử lý media

Đầu ra chính là các tệp phụ đề `.srt` và `.json`, cùng tệp âm thanh `.wav` đã trích xuất và chuẩn hóa.

### Nhìn nhanh

| Mục | Chi tiết |
|---|---|
| Điểm vào chính | `vad_lang_subtitle.py` |
| Input | Video/audio được FFmpeg hỗ trợ |
| Output | `*.wav`, `*.srt`, `*.json` |
| Luồng cốt lõi | VAD -> Whisper -> Lingua -> refinement |
| Trường hợp dùng điển hình | Tạo phụ đề đa ngôn ngữ |

---

## 🚀 Tính năng chính

- **Pipeline Silero VAD -> Whisper**  
  Voice Activity Detection (VAD) tách âm thanh thành các đoạn có tiếng nói, sau đó Whisper phiên âm từng đoạn.

- **Phát hiện ngôn ngữ chi tiết**  
  Dùng [Lingua](https://github.com/pemistahl/lingua-java) cùng bộ phát hiện của Whisper để gắn nhãn mọi đoạn (kể cả từng từ) bằng mã ngôn ngữ ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Tinh chỉnh đoạn thông minh**  
  Dọn timestamp để tránh hổng hoặc chồng lấn. Tách theo dấu câu để chia bản phiên âm dài tại dấu phẩy, dấu chấm, dấu hỏi, v.v. VAD merge căn lại các từ về block VAD để phụ đề mượt hơn. Phân đoạn theo độ dài áp dụng giới hạn riêng theo ngôn ngữ.

- **Phụ đề đa ngôn ngữ**  
  Xuất cả `.srt` và `.json`, giữ nhãn ngôn ngữ theo từng đoạn để bạn có thể style hoặc lọc theo ngôn ngữ trong player/editor downstream.

- **Xử lý media ổn định**  
  Tự động trích xuất và chuẩn hóa audio qua FFmpeg, thử sửa container lỗi, và áp dụng chuẩn hóa động (`dynaudnorm`) để bản phiên âm rõ hơn.

---

## 🔁 Luồng pipeline

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

Đường chạy chính trong `vad_lang_subtitle.py`:

1. Parse đối số CLI (`--video-path`, `--whisper-model`, `--force`).
2. Suy ra đường dẫn đầu ra từ basename đầu vào.
3. Trích xuất/chuẩn hóa âm thanh qua FFmpeg.
4. Tải Silero VAD (`torch.hub`) và model Whisper.
5. Phiên âm lượt đầu trên các chunk VAD.
6. Gộp/tinh chỉnh segment, rồi phiên âm lượt hai trên các đoạn đã gộp.
7. Áp dụng giảm độ dài phụ đề và dọn timestamp.
8. Lưu `.srt` và `.json`.

---

## 🗂 Cấu trúc dự án

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

> ⚠️ Lưu ý: README trước đó từng tham chiếu `requirements.txt`, nhưng hiện file này chưa có ở thư mục gốc repository.

---

## ✅ Điều kiện tiên quyết

- Python `3.10+` (đã kiểm tra trên môi trường 3.x hiện đại)
- Đã cài `ffmpeg` và có trong `PATH`
- CPU/GPU + RAM đủ cho model Whisper bạn chọn (với `large`, rất nên dùng GPU)
- Có Internet ở lần chạy đầu để tải trọng số model Whisper và tài nguyên Silero VAD (`torch.hub`)

Các package Python script đang dùng gồm:

- `torch`
- `torchaudio`
- `whisper` (OpenAI Whisper Python package)
- `lingua-language-detector`
- `tqdm`

Lệnh kiểm tra nhanh:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Cài đặt

1. **Clone repo này**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **Tạo và kích hoạt môi trường ảo**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Cài dependencies**

```bash
pip install -r requirements.txt
```

Nếu checkout của bạn vẫn chưa có `requirements.txt`, hãy cài thủ công các dependency runtime cốt lõi:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Và bảo đảm FFmpeg đã được cài ở cấp hệ thống.

---

## ⚡ Bắt đầu nhanh

Nếu bạn muốn đường ngắn nhất từ clone đến phụ đề:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Mẹo: dùng `small` khi lặp thử nhanh, rồi chuyển sang `large` cho chất lượng cuối cùng.

Các artifact dự kiến cạnh media đầu vào:

- `*.wav` audio đã trích xuất và chuẩn hóa
- `*.srt` tệp phụ đề cho player/editor
- `*.json` metadata phụ đề đa ngôn ngữ có cấu trúc

---

## 🎚 Hướng dẫn chọn model

Chọn model Whisper dựa trên mục tiêu tốc độ so với chất lượng:

| Model | Speed | Quality | Recommended Use |
|---|---|---|---|
| `tiny` / `base` | Fastest | Lowest | Fast smoke tests and pipeline validation |
| `small` | Fast | Good | Daily iteration and local development |
| `medium` | Medium | Better | Balanced production workflows |
| `large` (default) | Slowest | Best | Final subtitle exports for highest quality |

Mẫu triển khai thực tế:

1. Lặp với `small --force`
2. Xác thực timing và nhãn ngôn ngữ
3. Chạy lại với `large --force` để xuất bản giao cuối

---

## 🛠 Cách dùng

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Tùy chọn CLI

| Flag | Alias | Required | Description |
|---|---|---|---|
| `--video-path` | `-t` | Yes | Input media path (video/audio supported by FFmpeg) |
| `--whisper-model` | — | No | Whisper model name (default: `large`) |
| `--force` | — | No | Re-run even if `.wav`, `.srt`, or `.json` already exist |

### Hành vi xử lý

- Tên output được suy ra từ base path của input.
- Với `input.mp4`, output là `input.wav` (audio đã chuẩn hóa), `input.srt` (phụ đề có timestamp), và `input.json` (metadata gồm `start`, `end`, `lang`, `text`, và tùy chọn thời gian theo từ).
- Nếu `.srt` hoặc `.json` đã tồn tại thì sẽ skip trừ khi bật `--force`.

---

## ⚙️ Cấu hình

Cấu hình hiện tại chủ yếu theo CLI và mặc định trong code:

| Config Area | Current Behavior |
|---|---|
| Whisper model | `--whisper-model` (default `large`) |
| Processing sample rate | Hard-coded to `16000` for VAD/transcription processing |
| FFmpeg extraction | Mono WAV, `44100 Hz`, with `dynaudnorm=f=100` |
| Lingua detector | Initialized for `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` in main flow |
| Whisper-side filtering helper defaults | Includes `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Ghi chú giả định: danh sách ngôn ngữ ở helper defaults và cấu hình detector chính chưa hoàn toàn giống nhau; README này giữ đúng hành vi hiện đang được triển khai.

Chi tiết triển khai bổ sung từ script hiện tại:

- `torch.set_num_threads(1)` được áp dụng khi runtime.
- Model VAD được tải từ `snakers4/silero-vad` qua `torch.hub.load(...)`.
- Segment cleaning loại các mục có ngôn ngữ `und` hoặc văn bản rỗng.

---

## 📦 Định dạng đầu ra

Công cụ ghi hai artifact phụ đề cho mỗi media đầu vào:

- `*.srt`: Văn bản phụ đề chuẩn với timestamp `HH:MM:SS,mmm`.
- `*.json`: Danh sách phụ đề có cấu trúc, chứa timestamp đã định dạng và nhãn ngôn ngữ.

Dạng JSON segment điển hình:

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

Ghi chú:

- `start`/`end` được serialize thành chuỗi kiểu SRT trong output JSON.
- `words` có thể xuất hiện tùy theo giai đoạn xử lý/tinh chỉnh segment.
- Giá trị `lang` là `und` có thể xuất hiện ở các đoạn chưa chắc ngôn ngữ.

---

## 🧪 Ví dụ

Chạy với MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Chạy với MOV và ép ghi đè:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Chạy với input chỉ audio được FFmpeg hỗ trợ:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Ví dụ batch shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Ghi chú phát triển

- Script chính thức đang hoạt động là `vad_lang_subtitle.py`.
- Các file lịch sử (`*.old`, `*.shorterlength*`, `archived/`) hữu ích để tham khảo nhưng có vẻ không phải bản chuẩn.
- Hiện chưa có scaffold đóng gói dự án (`pyproject.toml`, `setup.py`) và chưa có CI/test suite được commit.
- `data/` chứa media mẫu dung lượng lớn; cần lưu ý kích thước repo và dung lượng đĩa local khi thử nghiệm.
- `clean_subtitles_dict()` có trong code nhưng hiện chưa được gọi ở pipeline chính.
- `--force` hiện là cơ chế để đảm bảo tái sinh output khi lặp tuning.

Vòng lặp dev local được gợi ý:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Dùng model nhỏ hơn (`tiny`/`base`/`small`) trong lúc lặp, rồi chuyển sang `large` cho chất lượng đầu ra cuối.

---

## 🩺 Khắc phục sự cố

| Symptom | What to do |
|---|---|
| `ffmpeg: command not found` | Cài FFmpeg và kiểm tra bằng `ffmpeg -version`. |
| First run is very slow or appears stuck | Lần đầu tải model (Whisper + Silero) có thể mất thời gian; các lần chạy sau sẽ nhanh hơn. |
| CUDA / GPU errors | Thử fallback CPU bằng model Whisper nhỏ hơn (`small`, `base`, `tiny`) và bảo đảm build PyTorch phù hợp môi trường. |
| Output files are not regenerated | Dùng `--force` để ghi đè các file đầu ra đã có. |
| `pip install -r requirements.txt` fails because file not found | Dùng lệnh cài dependency thủ công trong phần Installation. |
| Inaccurate language tagging on short segments | Có thể xảy ra ở đoạn cực ngắn/nhiễu; logic hiện kết hợp Whisper và Lingua nhưng vẫn có edge case. |
| Empty or near-empty subtitle output | Xác nhận input có tiếng nói, kiểm tra `.wav` đã trích xuất, rồi thử lại với `--force` sau khi xác thực bước FFmpeg. |
| Unexpected language flips between neighboring lines | Có thể xảy ra ở đoạn rất ngắn; cân nhắc hậu xử lý bằng cách gộp theo ngôn ngữ và thời lượng tối thiểu ở tooling downstream. |
| FFmpeg extraction fails on damaged media | Script sẽ thử lại sau bước sửa container (`-c copy -movflags +faststart`), nhưng file hỏng nặng vẫn có thể thất bại. |

Chẩn đoán nhanh:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Hạn chế và giả định đã biết

- Manifest dependency chưa được commit (`requirements.txt`, `pyproject.toml`, và `setup.py` không có ở repo root tại thời điểm viết).
- License được khai báo là MIT trong README, nhưng file `LICENSE` độc lập hiện chưa có.
- Lingua được khởi tạo rõ ràng với `EN/ZH/JA/AR` trong luồng chính, trong khi helper defaults gồm nhiều mã ứng viên hơn.
- Chưa có test/benchmark tự động được commit, nên việc xác thực hiện chủ yếu thủ công.
- Có các script lịch sử ở root và `archived/`; chỉ `vad_lang_subtitle.py` nên được xem là bản chính trừ khi bạn cố ý thử nghiệm.
- Script hiện in log runtime chi tiết và debug output theo từng segment; đây là hành vi dự kiến trong bản triển khai hiện tại.

---

## 🗺 Lộ trình

- Thêm và duy trì `requirements.txt` hoặc `pyproject.toml` có khóa phiên bản.
- Thêm test tự động cho logic phân đoạn và dọn timestamp.
- Thêm tài liệu benchmark và đánh giá chất lượng cho các ca biên đa ngôn ngữ.
- Thêm hỗ trợ file cấu hình thay vì chỉ dựa vào mặc định trong code.
- Mở rộng bộ README i18n trong `i18n/` và giữ đồng bộ language bar.
- Làm rõ và thống nhất hành vi chọn ngôn ngữ giữa cấu hình detector và helper defaults.
- Thêm file `LICENSE` chính thức để khớp khai báo trong README.

---

## 🔗 Lời cảm ơn

- [OpenAI Whisper](https://github.com/openai/whisper) cho chuyển giọng nói thành văn bản
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) cho phát hiện hoạt động giọng nói ổn định
- [Lingua](https://github.com/pemistahl/lingua-java) cho nhận diện ngôn ngữ độ chính xác cao

---

## 🤝 Đóng góp

1. Fork và clone
2. Tạo nhánh: `git checkout -b feat/your-idea`
3. Commit và push
4. Mở PR

Với thay đổi lớn, hãy kèm theo:

- Mô tả ngắn về thay đổi hành vi kỳ vọng
- Ví dụ lệnh có thể tái tạo
- Trích đoạn phụ đề trước/sau khi phù hợp

---

## 📫 Liên hệ

- Mở issue cho bug report, câu hỏi sử dụng, và đề xuất tính năng.
- Dùng các tùy chọn hỗ trợ ở trên cho tài trợ và câu hỏi về donation.

---

## 📄 Giấy phép

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
