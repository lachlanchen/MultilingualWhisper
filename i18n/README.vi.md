[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Trình tạo phụ đề có thể thay thế trực tiếp (drop-in), được xây dựng trên OpenAI Whisper và mở rộng với khả năng phát hiện, tinh chỉnh ngôn ngữ chính xác theo từng đoạn cho video chứa nhiều ngôn ngữ.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Tổng quan

`MultilingualWhisper` là một pipeline CLI Python xoay quanh [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Pipeline này kết hợp:

- Silero VAD để phân đoạn giọng nói
- OpenAI Whisper để phiên âm và dự đoán ngôn ngữ ban đầu
- Lingua để tinh chỉnh ngôn ngữ dựa trên văn bản
- FFmpeg để trích xuất, chuẩn hóa và xử lý media

Đầu ra chính là tệp phụ đề `.srt` và `.json`, cùng với tệp âm thanh `.wav` đã được trích xuất và chuẩn hóa.

### Tóm tắt nhanh

| Mục | Chi tiết |
|---|---|
| Điểm vào chính | `vad_lang_subtitle.py` |
| Đầu vào | Video/audio được FFmpeg hỗ trợ |
| Đầu ra | `*.wav`, `*.srt`, `*.json` |
| Luồng cốt lõi | VAD -> Whisper -> Lingua -> refinement |
| Trường hợp sử dụng điển hình | Tạo phụ đề đa ngôn ngữ |

---

## 🚀 Tính năng chính

- **Pipeline Silero VAD -> Whisper**  
  Voice Activity Detection (VAD) chia âm thanh thành các đoạn chứa tiếng nói, sau đó Whisper phiên âm từng đoạn.

- **Phát hiện ngôn ngữ chi tiết**  
  Sử dụng [Lingua](https://github.com/pemistahl/lingua-java) cùng với bộ phát hiện của Whisper để gắn mã ngôn ngữ ISO cho mọi đoạn (thậm chí từng từ riêng lẻ) như (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Tinh chỉnh đoạn thông minh**  
  Làm sạch mốc thời gian để tránh khoảng trống hoặc chồng lấn. Tách dấu câu để chia bản phiên âm dài tại dấu phẩy, dấu chấm, dấu hỏi, v.v. Gộp lại theo VAD để căn chỉnh từ về các khối VAD giúp phụ đề mượt hơn. Phân đoạn theo giới hạn độ dài phụ thuộc ngôn ngữ.

- **Phụ đề đa ngôn ngữ**  
  Xuất cả `.srt` và `.json`, giữ lại nhãn ngôn ngữ theo từng đoạn để bạn có thể style hoặc lọc theo ngôn ngữ trong player/trình chỉnh sửa ở bước sau.

- **Xử lý media mạnh mẽ**  
  Tự động trích xuất và chuẩn hóa âm thanh bằng FFmpeg, thử sửa container bị lỗi, và áp dụng chuẩn hóa động (`dynaudnorm`) để bản phiên âm rõ ràng hơn.

---

## 🗂 Cấu trúc dự án

```text
.
├── README.md
├── vad_lang_subtitle.py               # Pipeline chính: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old           # Prototype cũ
├── vad_lang_subtitle.py.20250706      # Snapshot lịch sử
├── vad_lang_subtitle.py.shorterlength # Biến thể lịch sử thay thế
├── vad_lang_subtitle.py.shorterlength2# Biến thể lịch sử thay thế
├── vad_lang_subtitle.srt              # Đầu ra ví dụ
├── vad_lang_subtitle.json             # JSON ví dụ
├── .github/
│   └── FUNDING.yml                    # Liên kết tài trợ
├── archived/                          # Thử nghiệm/prototype cũ
├── data/                              # Media mẫu tùy chọn + đầu ra đã tạo
├── figs/                              # Tài nguyên thương hiệu (banner/logo)
├── i18n/                              # Workspace bản dịch/readme (hiện có, đang trống)
└── .auto-readme-work/                 # Artefact workspace tạo README
```

> ⚠️ Lưu ý: README trước đây có nhắc đến `requirements.txt`, nhưng hiện tại tệp này không có ở thư mục gốc của repository.

---

## ✅ Điều kiện tiên quyết

- Python `3.10+` (đã kiểm thử với các môi trường 3.x hiện đại)
- `ffmpeg` đã cài và có trong `PATH`
- Đủ CPU/GPU + RAM cho model Whisper được chọn (với `large`, nên dùng GPU)
- Có Internet ở lần chạy đầu để tải trọng số model Whisper và tài nguyên Silero VAD (`torch.hub`)

Các gói Python được script sử dụng gồm:

- `torch`
- `torchaudio`
- `whisper` (gói Python OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

---

## 🔧 Cài đặt

1. **Clone repository**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
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

Nếu `requirements.txt` vẫn chưa có trong bản checkout của bạn, hãy cài thủ công các dependency runtime cốt lõi:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Đồng thời đảm bảo FFmpeg được cài ở cấp hệ thống.

---

## 🛠 Cách dùng

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Tùy chọn CLI

| Flag | Alias | Bắt buộc | Mô tả |
|---|---|---|---|
| `--video-path` | `-t` | Có | Đường dẫn media đầu vào (video/audio được FFmpeg hỗ trợ) |
| `--whisper-model` | — | Không | Tên model Whisper (mặc định: `large`) |
| `--force` | — | Không | Chạy lại kể cả khi `.wav`, `.srt`, hoặc `.json` đã tồn tại |

### Hành vi xử lý

- Tên tệp đầu ra được suy ra từ tên gốc của tệp đầu vào.
- Với `input.mp4`, đầu ra sẽ là `input.wav` (âm thanh đã chuẩn hóa), `input.srt` (phụ đề có mốc thời gian), và `input.json` (metadata gồm `start`, `end`, `lang`, `text`, có thể kèm mốc theo từ).
- Nếu `.srt` hoặc `.json` đã tồn tại thì sẽ bỏ qua, trừ khi bật `--force`.

---

## ⚙️ Cấu hình

Cấu hình hiện tại chủ yếu được điều khiển bằng CLI và giá trị mặc định trong mã:

- Whisper model: `--whisper-model` (mặc định `large`)
- Tần số lấy mẫu: hard-code `16000` cho xử lý
- Trích xuất FFmpeg: WAV mono, `44100 Hz`, với `dynaudnorm=f=100`
- Bộ phát hiện Lingua: khởi tạo cho `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` trong luồng chính
- Danh sách mã ngôn ngữ cho bước lọc phía Whisper trong helper defaults gồm `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`

Ghi chú giả định: danh sách ngôn ngữ trong helper defaults và trong cấu hình bộ phát hiện chính chưa hoàn toàn giống nhau; README này giữ nguyên hành vi hiện tại theo đúng phần đã triển khai.

---

## 🧪 Ví dụ

Chạy với MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Chạy với MOV và ghi đè cưỡng bức:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Chạy với đầu vào chỉ âm thanh được FFmpeg hỗ trợ:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 Ghi chú phát triển

- Script chính đang hoạt động là `vad_lang_subtitle.py`.
- Các tệp lịch sử (`*.old`, `*.shorterlength*`, `archived/`) hữu ích để tham khảo nhưng có vẻ không phải bản chuẩn.
- Hiện chưa có scaffolding dạng package (`pyproject.toml`, `setup.py`) và chưa có bộ CI/test được commit.
- `data/` chứa các artefact media mẫu có dung lượng lớn; cần lưu ý kích thước repository và dung lượng đĩa cục bộ khi thử nghiệm.
- `clean_subtitles_dict()` có tồn tại trong code nhưng hiện chưa được gọi trong pipeline chính.

---

## 🩺 Khắc phục sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `ffmpeg: command not found` | Cài FFmpeg và kiểm tra bằng `ffmpeg -version`. |
| Lần chạy đầu rất chậm hoặc có vẻ bị treo | Tải model lần đầu (Whisper + Silero) có thể mất thời gian; các lần chạy lại sẽ nhanh hơn. |
| Lỗi CUDA / GPU | Thử fallback CPU bằng model Whisper nhỏ hơn (`small`, `base`, `tiny`) và đảm bảo bản build PyTorch phù hợp với môi trường của bạn. |
| Tệp đầu ra không được tạo lại | Dùng `--force` để ghi đè tệp đầu ra đã tồn tại. |
| `pip install -r requirements.txt` lỗi do không tìm thấy tệp | Dùng lệnh cài dependency thủ công trong phần Cài đặt. |
| Gắn nhãn ngôn ngữ không chính xác trên đoạn quá ngắn | Có thể xảy ra với đoạn cực ngắn/nhiễu; logic hiện tại kết hợp Whisper và Lingua nhưng vẫn còn edge case. |

---

## 🗺 Lộ trình

- Thêm và duy trì `requirements.txt` hoặc `pyproject.toml` có pin phiên bản.
- Thêm kiểm thử tự động cho logic phân đoạn và làm sạch mốc thời gian.
- Bổ sung tài liệu benchmark và đánh giá chất lượng cho các trường hợp đa ngôn ngữ khó.
- Thêm hỗ trợ tệp cấu hình tùy chọn thay cho hành vi chỉ dựa trên mặc định trong mã.
- Mở rộng bộ README i18n trong `i18n/` và giữ đồng bộ thanh chọn ngôn ngữ.

---

## 💖 Hỗ trợ

Nếu dự án này hữu ích với bạn, bạn có thể hỗ trợ phát triển qua:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Trang cá nhân: https://lazying.art
- Chat/cộng đồng: https://chat.lazying.art
- Trung tâm ý tưởng/dự án: https://onlyideas.art

---

## 🔗 Lời cảm ơn

- [OpenAI Whisper](https://github.com/openai/whisper) cho speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) cho voice activity detection mạnh mẽ
- [Lingua](https://github.com/pemistahl/lingua-java) cho nhận diện ngôn ngữ độ chính xác cao

---

## 🤝 Đóng góp

1. Fork và clone
2. Tạo nhánh: `git checkout -b feat/your-idea`
3. Commit và push
4. Mở PR

---

## 📄 Giấy phép

MIT © Lachlan Chen
