[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Một công cụ tạo phụ đề kiểu gắn trực tiếp, xây dựng trên OpenAI Whisper, mở rộng thêm khả năng phát hiện và tinh chỉnh ngôn ngữ chi tiết theo từng đoạn cho các video có nhiều ngôn ngữ.

> Tạo phụ đề đa ngôn ngữ sạch hơn từ dữ liệu âm/video thực tế có nội dung hỗn hợp ngôn ngữ với phân đoạn theo ngôn ngữ.

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

| Mục tiêu | Giá trị |
| --- | --- |
| Đầu vào | Âm thanh/video tương thích FFmpeg |
| Quy trình | VAD segmentation → Whisper transcription → Lingua refinement |
| Đầu ra | `*.wav`, `*.srt`, và `*.json` đã chuẩn hóa |
| Trường hợp phù hợp | Phụ đề đa ngôn ngữ có nhãn ngôn ngữ theo từng đoạn |

---

## Mục lục

- [Tổng quan](#-tổng-quan)
- [Nhìn tổng quát](#nhìn-tổng-quát)
- [Tính năng chính](#-tính-năng-chính)
- [Luồng pipeline](#-luồng-pipeline)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Điều kiện tiên quyết](#-điều-kiện-tiên-quyết)
- [Cài đặt](#-cài-đặt)
- [Bắt đầu nhanh](#-bắt-đầu-nhanh)
- [Cách dùng](#-cách-dùng)
- [Cấu hình](#-cấu-hình)
- [Định dạng đầu ra](#-định-dạng-đầu-ra)
- [Ví dụ](#-ví-dụ)
- [Ghi chú phát triển](#-ghi-chú-phát-triển)
- [Khắc phục sự cố](#-khắc-phục-sự-cố)
- [Hạn chế và giả định đã biết](#-hạn-chế-và-giả-định-đã-biết)
- [Lộ trình](#-lộ-trình)
- [Hỗ trợ](#-hỗ-trợ)
- [Liên hệ](#-liên-hệ)
- [Lời cảm ơn](#-lời-cảm-ơn)
- [Đóng góp](#-đóng-góp)
- [Giấy phép](#-giấy-phép)

---

## ✨ Tổng quan

`MultilingualWhisper` là pipeline CLI Python tập trung tại [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Nó kết hợp:

- Silero VAD để phân đoạn tiếng nói
- OpenAI Whisper để phiên âm và dự đoán ngôn ngữ ban đầu
- Lingua để tinh chỉnh ngôn ngữ từ văn bản
- FFmpeg để trích xuất, chuẩn hóa và xử lý media

Đầu ra chính là các tệp phụ đề `.srt` và `.json`, cùng tệp âm thanh `.wav` đã trích xuất và chuẩn hóa.

### Nhìn nhanh

| Mục | Chi tiết |
|---|---|
| Điểm vào chính | `vad_lang_subtitle.py` |
| Đầu vào | Video/audio được FFmpeg hỗ trợ |
| Đầu ra | `*.wav`, `*.srt`, `*.json` |
| Luồng cốt lõi | VAD -> Whisper -> Lingua -> tinh chỉnh |
| Trường hợp dùng điển hình | Tạo phụ đề đa ngôn ngữ |

---

## 🚀 Tính năng chính

- **Pipeline Silero VAD -> Whisper**
  Voice Activity Detection (VAD) chia âm thanh thành các đoạn có tiếng nói, sau đó Whisper phiên âm từng đoạn.

- **Phát hiện ngôn ngữ chi tiết**
  Sử dụng [Lingua](https://github.com/pemistahl/lingua-java) cùng detector của Whisper để gắn nhãn cho mọi đoạn (kể cả từng từ) bằng mã ngôn ngữ ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Tinh chỉnh phân đoạn thông minh**
  Dọn timestamp để tránh khoảng trống hoặc chồng chéo. Phân tách dấu câu cắt các đoạn phiên âm dài tại dấu phẩy, chấm, dấu hỏi, v.v. VAD sẽ gộp lại và căn chỉnh từ vào các khối VAD để phụ đề mượt hơn. Tách theo độ dài áp dụng giới hạn riêng cho từng ngôn ngữ.

- **Phụ đề đa ngôn ngữ**
  Xuất cả `.srt` và `.json`, giữ nhãn ngôn ngữ theo từng đoạn để bạn có thể tùy chỉnh kiểu dáng hoặc lọc theo ngôn ngữ ở trình phát hoặc trình chỉnh sửa phía sau.

- **Xử lý media ổn định**
  Tự động trích xuất và chuẩn hóa âm thanh qua FFmpeg, cố gắng sửa các container lỗi, và áp dụng chuẩn hóa động (`dynaudnorm`) để bản phiên âm rõ ràng hơn.

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

1. Phân tích tham số CLI (`--video-path`, `--whisper-model`, `--force`).
2. Xác định đường dẫn đầu ra từ basename đầu vào.
3. Trích xuất/chuẩn hóa âm thanh qua FFmpeg.
4. Tải Silero VAD (`torch.hub`) và mô hình Whisper.
5. Phiên âm lần đầu trên từng đoạn VAD.
6. Gộp/tinh chỉnh đoạn, rồi phiên âm lần hai trên các đoạn đã gộp.
7. Áp dụng giảm độ dài phụ đề và làm sạch timestamp.
8. Lưu `.srt` và `.json`.

---

## 🗂 Cấu trúc dự án

```text
.
├── README.md
├── vad_lang_subtitle.py                # Main pipeline: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old            # Bản dựng mẫu cũ
├── vad_lang_subtitle.py.20250706       # Snapshot lịch sử
├── vad_lang_subtitle.py.shorterlength  # Biến thể lịch sử thay thế
├── vad_lang_subtitle.py.shorterlength2 # Biến thể lịch sử thay thế
├── vad_lang_subtitle.srt               # Ví dụ đầu ra
├── vad_lang_subtitle.json              # Ví dụ JSON
├── .github/
│   └── FUNDING.yml                     # Liên kết tài trợ
├── archived/
│   ├── vad.py
│   ├── vad_lang.py
│   ├── vad_lang_subtitle.py
│   ├── decode_audio.py
│   ├── decode_audio_v2.py
│   ├── text_language_detect.py
│   └── trans_with_lang.py
├── data/                               # Media mẫu + đầu ra sinh
├── figs/                               # Tài nguyên nhận diện thương hiệu (banner/logo)
├── i18n/                               # Các README đa ngôn ngữ hiện có
└── .auto-readme-work/                  # Dữ liệu hỗ trợ tạo README
```

> ⚠️ Ghi chú: README tiếng Anh trước đó tham chiếu `requirements.txt`, nhưng hiện file này chưa có trong repo root.

---

## ✅ Điều kiện tiên quyết

- Python `3.10+` (đã kiểm tra trên môi trường Python 3.x hiện đại)
- Có cài `ffmpeg` và có trong `PATH`
- CPU/GPU + RAM đủ cho mô hình Whisper đã chọn (với `large`, GPU được khuyến nghị mạnh)
- Có kết nối internet khi chạy lần đầu để tải trọng số Whisper và tài nguyên Silero VAD (`torch.hub`)

Các gói Python được script dùng:

- `torch`
- `torchaudio`
- `whisper` (gói Python của OpenAI Whisper)
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

3. **Cài đặt dependencies**

```bash
pip install -r requirements.txt
```

Nếu `requirements.txt` vẫn chưa có trong checkout của bạn, cài trực tiếp các dependency runtime cốt lõi:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Và đảm bảo FFmpeg đã được cài ở cấp hệ thống.

---

## ⚡ Bắt đầu nhanh

Nếu bạn muốn đường đi nhanh nhất từ clone đến phụ đề:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Mẹo: dùng `small` khi đang thử nghiệm, sau đó chuyển sang `large` cho chất lượng cuối cùng.

Kết quả dự kiến nằm cạnh media đầu vào:

- `*.wav` âm thanh đã trích xuất và chuẩn hóa
- `*.srt` tệp phụ đề cho trình phát/trình chỉnh sửa
- `*.json` metadata phụ đề đa ngôn ngữ có cấu trúc

---

## 🛠 Cách dùng

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Tùy chọn CLI

| Tham số | Bí danh | Bắt buộc | Mô tả |
|---|---|---|---|
| `--video-path` | `-t` | Có | Đường dẫn media đầu vào (video/audio được FFmpeg hỗ trợ) |
| `--whisper-model` | — | Không | Tên mô hình Whisper (mặc định: `large`) |
| `--force` | — | Không | Chạy lại dù đã có `.wav`, `.srt`, hoặc `.json` |

### Hành vi xử lý

- Tên đầu ra được suy ra từ đường dẫn cơ sở của input.
- Với `input.mp4`, output là `input.wav` (âm thanh đã chuẩn hóa), `input.srt` (phụ đề có timestamp), và `input.json` (metadata bao gồm `start`, `end`, `lang`, `text`, và tùy chọn word timings).
- Các file `.srt` hoặc `.json` đã có sẽ bị bỏ qua trừ khi bật `--force`.

---

## ⚙️ Cấu hình

Cấu hình hiện tại chủ yếu được điều khiển qua CLI và mặc định trong code:

| Khu vực cấu hình | Hành vi hiện tại |
|---|---|
| Mô hình Whisper | `--whisper-model` (mặc định `large`) |
| Tần số mẫu xử lý | Cố định `16000` cho xử lý VAD/phiên âm |
| Trích xuất FFmpeg | WAV mono, `44100 Hz`, với `dynaudnorm=f=100` |
| Bộ phát hiện Lingua | Khởi tạo cho `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` trong luồng chính |
| Mặc định lọc bên phía Whisper | Bao gồm `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Ghi chú giả định: danh sách ngôn ngữ trong mặc định helper và thiết lập detector chính chưa hoàn toàn trùng khớp; README này giữ đúng hành vi hiện tại.

---

## 📦 Định dạng đầu ra

Công cụ tạo hai tệp phụ đề cho mỗi media đầu vào:

- `*.srt`: Văn bản phụ đề chuẩn với timestamp `HH:MM:SS,mmm`.
- `*.json`: Danh sách phụ đề có cấu trúc chứa timestamp đã định dạng và nhãn ngôn ngữ.

Dạng JSON điển hình:

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

- `start`/`end` được serialize dưới dạng chuỗi kiểu SRT trong JSON.
- `words` có thể xuất hiện tùy vào giai đoạn xử lý/tinh chỉnh của segment.
- Giá trị `lang` là `und` có thể xuất hiện cho đoạn không chắc chắn về ngôn ngữ.

---

## 🧪 Ví dụ

Chạy trên một file MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Chạy trên file MOV và ép ghi đè:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Chạy trên input chỉ âm thanh được FFmpeg hỗ trợ:

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

- Script đang hoạt động chuẩn là `vad_lang_subtitle.py`.
- Các file lịch sử (`*.old`, `*.shorterlength*`, `archived/`) hữu ích để tham khảo nhưng hiện không phải bản chính.
- Hiện chưa có scaffold dự án đóng gói (`pyproject.toml`, `setup.py`) và chưa có CI/test suite.
- `data/` chứa media mẫu lớn; chú ý đến dung lượng repo và dung lượng đĩa cục bộ khi chạy thử.
- `clean_subtitles_dict()` tồn tại trong code nhưng hiện chưa được gọi trong pipeline chính.
- `--force` hiện là cơ chế để buộc sinh lại output cho mỗi lần tuning.

Gợi ý luồng dev local:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Dùng mô hình nhỏ hơn (`tiny`/`base`/`small`) khi đang thử nghiệm, rồi chuyển sang `large` cho chất lượng cuối.

---

## 🩺 Khắc phục sự cố

| Triệu chứng | Cần làm gì |
|---|---|
| `ffmpeg: command not found` | Cài FFmpeg và kiểm tra lại bằng `ffmpeg -version`. |
| Chạy lần đầu rất chậm hoặc có vẻ bị treo | Việc tải ban đầu của model (Whisper + Silero) có thể mất thời gian; lần chạy lại sẽ nhanh hơn. |
| Lỗi CUDA / GPU | Thử fallback CPU bằng mô hình Whisper nhỏ hơn (`small`, `base`, `tiny`) và đảm bảo build PyTorch phù hợp với môi trường của bạn. |
| Tệp đầu ra không được tái sinh | Dùng `--force` để ghi đè các tệp đã có. |
| `pip install -r requirements.txt` lỗi vì không tìm thấy file | Dùng lệnh cài dependencies thủ công như đã nêu trong phần Cài đặt. |
| Gắn nhãn ngôn ngữ không chính xác trên đoạn ngắn | Trường hợp này có thể xảy ra với các đoạn rất ngắn hoặc nhiễu; logic hiện tại kết hợp Whisper và Lingua nhưng vẫn có trường hợp biên. |
| Kết quả phụ đề rỗng hoặc gần rỗng | Kiểm tra input có tiếng nói chưa, kiểm tra `.wav` đã trích xuất, rồi chạy lại với `--force` sau khi xác thực bước FFmpeg. |
| Ngôn ngữ đổi đột ngột giữa hai dòng kế tiếp | Có thể xảy ra với đoạn rất ngắn; cân nhắc gộp lại trong công cụ downstream theo ngôn ngữ và thời lượng tối thiểu. |

---

## ⚠️ Hạn chế và giả định đã biết

- Manifest dependency chưa được commit (`requirements.txt`, `pyproject.toml`, và `setup.py` đều vắng mặt tại repository root thời điểm viết).
- License được khai báo trong README là MIT, nhưng file `LICENSE` độc lập hiện chưa có.
- Lingua được khởi tạo rõ ràng với `EN/ZH/JA/AR` trong luồng chính, trong khi mặc định helper bao gồm nhiều mã ứng viên hơn.
- Hiện chưa có test/benchmark tự động được commit, nên xác thực chủ yếu thủ công.
- Script lịch sử có mặt ở root và `archived/`; chỉ `vad_lang_subtitle.py` được coi là bản đang dùng, trừ khi bạn đang thử nghiệm có chủ đích.

---

## 🗺 Lộ trình

- Thêm và duy trì `requirements.txt` hoặc `pyproject.toml` đã khóa phiên bản.
- Thêm tests tự động cho logic phân đoạn và dọn timestamp.
- Thêm tài liệu benchmark và đánh giá chất lượng cho các trường hợp đa ngôn ngữ biên.
- Thêm hỗ trợ file cấu hình thay vì chỉ dùng mặc định trong code.
- Mở rộng bộ README i18n trong `i18n/` và đồng bộ thanh ngôn ngữ.
- Rõ ràng hóa và thống nhất hành vi chọn ngôn ngữ giữa cấu hình detector và defaults.
- Thêm file `LICENSE` chính thức đúng theo thông báo trong README.

---

## 🔗 Lời cảm ơn

- [OpenAI Whisper](https://github.com/openai/whisper) cho chuyển thoại thành văn bản
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) cho phát hiện hoạt động giọng nói đáng tin cậy
- [Lingua](https://github.com/pemistahl/lingua-java) cho nhận diện ngôn ngữ độ chính xác cao

---

## 🤝 Đóng góp

1. Fork và clone
2. Tạo nhánh: `git checkout -b feat/your-idea`
3. Commit và push
4. Mở PR

Với thay đổi lớn, hãy kèm:

- Mô tả ngắn gọn về thay đổi hành vi dự kiến
- Ví dụ lệnh tái tạo có thể thực hiện
- Đoạn phụ đề trước/sau khi có liên quan

---

## 📫 Liên hệ

- Mở issue để báo lỗi, hỏi cách dùng, và gửi đề xuất tính năng.
- Dùng các tùy chọn hỗ trợ ở trên cho tài trợ và các thắc mắc về donation.

---

## 📄 Giấy phép

MIT © Lachlan Chen


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
