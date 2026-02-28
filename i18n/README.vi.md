[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Một trình tạo phụ đề thay thế trực tiếp, xây dựng trên OpenAI Whisper, được mở rộng với khả năng phát hiện ngôn ngữ chính xác theo từng đoạn và tinh chỉnh cho video chứa nhiều ngôn ngữ.

> Tạo phụ đề đa ngôn ngữ sạch hơn từ media thực tế có trộn nhiều ngôn ngữ, nhờ cơ chế phân đoạn theo ngôn ngữ.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)

---

## Mục lục

- [Tổng quan](#-tổng-quan)
- [Nhìn nhanh](#nhìn-nhanh)
- [Tính năng chính](#-tính-năng-chính)
- [Luồng xử lý](#-luồng-xử-lý)
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
- [Lời cảm ơn](#-lời-cảm-ơn)
- [Đóng góp](#-đóng-góp)
- [Giấy phép](#-giấy-phép)

---

## ✨ Tổng quan

`MultilingualWhisper` là một pipeline CLI Python tập trung quanh [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Công cụ kết hợp:

- Silero VAD để phân đoạn tiếng nói
- OpenAI Whisper để chép lời và dự đoán ngôn ngữ ban đầu
- Lingua để tinh chỉnh ngôn ngữ dựa trên văn bản
- FFmpeg để trích xuất, chuẩn hóa và xử lý media

Đầu ra chính là các tệp phụ đề `.srt` và `.json`, cùng tệp âm thanh `.wav` đã trích xuất và chuẩn hóa.

### Nhìn nhanh

| Mục | Chi tiết |
|---|---|
| Điểm vào chính | `vad_lang_subtitle.py` |
| Đầu vào | Video/âm thanh được FFmpeg hỗ trợ |
| Đầu ra | `*.wav`, `*.srt`, `*.json` |
| Luồng cốt lõi | VAD -> Whisper -> Lingua -> tinh chỉnh |
| Trường hợp dùng điển hình | Tạo phụ đề đa ngôn ngữ |

---

## 🚀 Tính năng chính

- **Pipeline Silero VAD -> Whisper**
  Voice Activity Detection (VAD) chia âm thanh thành các đoạn tiếng nói, rồi Whisper chép lời từng đoạn.

- **Phát hiện ngôn ngữ chi tiết**
  Dùng [Lingua](https://github.com/pemistahl/lingua-java) cùng bộ phát hiện của Whisper để gắn nhãn cho mọi đoạn (kể cả từng từ) bằng mã ngôn ngữ ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Tinh chỉnh đoạn thông minh**
  Làm sạch timestamp để tránh khoảng trống hoặc chồng lấn. Tách theo dấu câu để cắt bản chép dài tại dấu phẩy, chấm, hỏi, v.v. Gộp theo VAD để căn từ về các khối VAD cho phụ đề mượt hơn. Phân đoạn theo độ dài áp dụng giới hạn riêng theo từng ngôn ngữ.

- **Phụ đề đa ngôn ngữ**
  Xuất đồng thời `.srt` và `.json`, giữ nhãn ngôn ngữ theo từng đoạn để bạn có thể tùy biến hiển thị hoặc lọc theo ngôn ngữ ở player/trình biên tập phía sau.

- **Xử lý media bền vững**
  Tự động trích xuất và chuẩn hóa âm thanh bằng FFmpeg, cố gắng sửa container lỗi, và áp dụng chuẩn hóa động (`dynaudnorm`) để bản chép rõ hơn.

---

## 🔁 Luồng xử lý

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

Luồng chạy chính trong `vad_lang_subtitle.py`:

1. Phân tích tham số CLI (`--video-path`, `--whisper-model`, `--force`).
2. Suy ra đường dẫn đầu ra từ basename của đầu vào.
3. Trích xuất/chuẩn hóa âm thanh bằng FFmpeg.
4. Tải Silero VAD (`torch.hub`) và mô hình Whisper.
5. Chép lời lượt đầu trên các đoạn VAD.
6. Gộp/tinh chỉnh đoạn, rồi chép lời lượt hai trên các khoảng đã gộp.
7. Áp dụng rút gọn độ dài phụ đề và làm sạch timestamp.
8. Lưu `.srt` và `.json`.

---

## 🗂 Cấu trúc dự án

```text
.
├── README.md
├── vad_lang_subtitle.py                # Pipeline chính: VAD -> Whisper -> Lingua -> tinh chỉnh -> lưu
├── vad_lang_subtitle.py.old            # Bản prototype cũ
├── vad_lang_subtitle.py.20250706       # Bản chụp lịch sử
├── vad_lang_subtitle.py.shorterlength  # Biến thể lịch sử thay thế
├── vad_lang_subtitle.py.shorterlength2 # Biến thể lịch sử thay thế
├── vad_lang_subtitle.srt               # Đầu ra mẫu
├── vad_lang_subtitle.json              # JSON mẫu
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
├── data/                               # Media mẫu tùy chọn + đầu ra đã tạo
├── figs/                               # Tài nguyên thương hiệu (banner/logo)
├── i18n/                               # Các tệp README đa ngôn ngữ hiện có
└── .auto-readme-work/                  # Tạo phẩm không gian làm việc của quá trình sinh README
```

> ⚠️ Lưu ý: README trước đây có nhắc đến `requirements.txt`, nhưng hiện tại tệp này không có ở thư mục gốc repository.

---

## ✅ Điều kiện tiên quyết

- Python `3.10+` (đã kiểm tra trên môi trường 3.x hiện đại)
- `ffmpeg` đã cài đặt và có trên `PATH`
- CPU/GPU + RAM đủ cho mô hình Whisper bạn chọn (với `large`, rất nên dùng GPU)
- Có Internet ở lần chạy đầu để tải trọng số mô hình Whisper và tài nguyên Silero VAD (`torch.hub`)

Các gói Python script đang dùng gồm:

- `torch`
- `torchaudio`
- `whisper` (gói Python OpenAI Whisper)
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

Nếu bản checkout của bạn vẫn chưa có `requirements.txt`, hãy cài thủ công các dependency runtime cốt lõi:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Và đảm bảo FFmpeg đã được cài ở cấp hệ thống.

---

## ⚡ Bắt đầu nhanh

Nếu bạn muốn đi từ clone đến phụ đề theo cách nhanh nhất:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Mẹo: dùng `small` khi lặp nhanh, sau đó chuyển sang `large` cho chất lượng cuối cùng.

Các tệp tạo ra dự kiến bên cạnh media đầu vào:

- `*.wav` âm thanh trích xuất đã chuẩn hóa
- `*.srt` tệp phụ đề cho player/trình biên tập
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

| Flag | Alias | Bắt buộc | Mô tả |
|---|---|---|---|
| `--video-path` | `-t` | Có | Đường dẫn media đầu vào (video/âm thanh được FFmpeg hỗ trợ) |
| `--whisper-model` | — | Không | Tên mô hình Whisper (mặc định: `large`) |
| `--force` | — | Không | Chạy lại kể cả khi `.wav`, `.srt`, hoặc `.json` đã tồn tại |

### Hành vi xử lý

- Tên đầu ra được suy ra từ đường dẫn gốc của đầu vào.
- Với `input.mp4`, đầu ra là `input.wav` (âm thanh đã chuẩn hóa), `input.srt` (phụ đề có timestamp), và `input.json` (metadata gồm `start`, `end`, `lang`, `text`, tùy chọn thời gian theo từ).
- Nếu `.srt` hoặc `.json` đã tồn tại thì sẽ bỏ qua, trừ khi đặt `--force`.

---

## ⚙️ Cấu hình

Cấu hình hiện tại chủ yếu được điều khiển bởi CLI và giá trị mặc định trong mã:

| Khu vực cấu hình | Hành vi hiện tại |
|---|---|
| Mô hình Whisper | `--whisper-model` (mặc định `large`) |
| Tần số lấy mẫu xử lý | Hard-code `16000` cho xử lý VAD/chép lời |
| Trích xuất FFmpeg | WAV mono, `44100 Hz`, với `dynaudnorm=f=100` |
| Bộ phát hiện Lingua | Khởi tạo cho `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` trong luồng chính |
| Mặc định helper lọc phía Whisper | Gồm `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Ghi chú giả định: danh sách ngôn ngữ trong mặc định helper và cấu hình bộ phát hiện chính chưa hoàn toàn đồng nhất; README này giữ nguyên hành vi hiện được triển khai.

---

## 📦 Định dạng đầu ra

Công cụ ghi hai tạo phẩm phụ đề cho mỗi media đầu vào:

- `*.srt`: Văn bản phụ đề chuẩn với timestamp `HH:MM:SS,mmm`.
- `*.json`: Danh sách phụ đề có cấu trúc, chứa timestamp đã định dạng và nhãn ngôn ngữ.

Dạng segment JSON điển hình:

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

Lưu ý:

- `start`/`end` được tuần tự hóa thành chuỗi kiểu SRT trong đầu ra JSON.
- `words` có thể xuất hiện tùy theo giai đoạn xử lý/tinh chỉnh segment.
- Giá trị `lang` là `und` có thể xuất hiện với những đoạn không chắc chắn về ngôn ngữ.

---

## 🧪 Ví dụ

Chạy với một tệp MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Chạy với MOV và buộc ghi đè:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Chạy với đầu vào chỉ âm thanh được FFmpeg hỗ trợ:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Ví dụ chạy hàng loạt bằng shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Ghi chú phát triển

- Script chuẩn hiện tại là `vad_lang_subtitle.py`.
- Các tệp lịch sử (`*.old`, `*.shorterlength*`, `archived/`) hữu ích để tham khảo nhưng có vẻ không phải bản chuẩn.
- Hiện chưa có khung đóng gói dự án (`pyproject.toml`, `setup.py`) và chưa commit bộ CI/test.
- `data/` chứa các tạo phẩm media mẫu lớn; cần lưu ý kích thước repository và dung lượng đĩa cục bộ khi thử nghiệm.
- `clean_subtitles_dict()` có trong mã nhưng hiện không được gọi trong pipeline chính.
- `--force` là cơ chế hiện tại để đảm bảo tái tạo đầu ra khi tinh chỉnh lặp.

Vòng lặp phát triển cục bộ được đề xuất:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Dùng mô hình nhỏ hơn (`tiny`/`base`/`small`) khi lặp nhanh, sau đó chuyển sang `large` để có chất lượng đầu ra cuối cùng.

---

## 🩺 Khắc phục sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `ffmpeg: command not found` | Cài FFmpeg và kiểm tra bằng `ffmpeg -version`. |
| Lần chạy đầu rất chậm hoặc có vẻ bị treo | Tải mô hình lần đầu (Whisper + Silero) có thể mất thời gian; các lần chạy lại sẽ nhanh hơn. |
| Lỗi CUDA / GPU | Thử chạy CPU bằng mô hình Whisper nhỏ hơn (`small`, `base`, `tiny`) và đảm bảo bản dựng PyTorch phù hợp với môi trường của bạn. |
| Tệp đầu ra không được tạo lại | Dùng `--force` để ghi đè các tệp đầu ra đã tồn tại. |
| `pip install -r requirements.txt` lỗi vì không tìm thấy tệp | Dùng lệnh cài dependency thủ công trong mục Cài đặt. |
| Gắn nhãn ngôn ngữ không chính xác trên đoạn ngắn | Điều này có thể xảy ra với đoạn cực ngắn/nhiễu; logic hiện tại kết hợp Whisper và Lingua nhưng vẫn có trường hợp biên. |
| Đầu ra phụ đề trống hoặc gần như trống | Xác nhận đầu vào có tiếng nói, kiểm tra `.wav` đã trích xuất, rồi thử lại với `--force` sau khi xác thực bước trích xuất FFmpeg. |
| Ngôn ngữ thay đổi bất thường giữa các dòng liền kề | Có thể xảy ra ở đoạn rất ngắn; cân nhắc gộp hậu xử lý ở công cụ downstream theo ngôn ngữ và thời lượng tối thiểu. |

---

## ⚠️ Hạn chế và giả định đã biết

- Chưa commit tệp khai báo dependency (`requirements.txt`, `pyproject.toml`, và `setup.py` đều không có ở thư mục gốc tại thời điểm viết).
- Giấy phép được khai báo là MIT trong README, nhưng hiện chưa có tệp `LICENSE` riêng.
- Lingua được khởi tạo tường minh với `EN/ZH/JA/AR` trong luồng chính, trong khi mặc định helper gồm nhiều mã ứng viên hơn.
- Hiện chưa commit test/benchmark tự động, nên việc xác thực chủ yếu là thủ công.
- Có các script lịch sử ở thư mục gốc và `archived/`; chỉ nên xem `vad_lang_subtitle.py` là bản hoạt động chính trừ khi bạn chủ đích thử nghiệm.

---

## 🗺 Lộ trình

- Thêm và duy trì `requirements.txt` hoặc `pyproject.toml` có ghim phiên bản.
- Thêm test tự động cho logic phân đoạn và làm sạch timestamp.
- Thêm tài liệu benchmark và đánh giá chất lượng cho các trường hợp biên đa ngôn ngữ.
- Thêm hỗ trợ tệp cấu hình tùy chọn thay cho hành vi chỉ dựa vào mặc định trong mã.
- Mở rộng bộ README i18n trong `i18n/` và giữ đồng bộ thanh chuyển ngôn ngữ.
- Làm rõ và thống nhất hành vi chọn ngôn ngữ giữa cấu hình detector và mặc định helper.
- Thêm tệp `LICENSE` chính thức để khớp với khai báo trong README.

---

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

Liên kết hỗ trợ/cộng đồng bổ sung:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Trang cá nhân: https://lazying.art
- Chat/cộng đồng: https://chat.lazying.art
- Trung tâm ý tưởng/dự án: https://onlyideas.art

---

## 🔗 Lời cảm ơn

- [OpenAI Whisper](https://github.com/openai/whisper) cho speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) cho voice activity detection ổn định
- [Lingua](https://github.com/pemistahl/lingua-java) cho nhận diện ngôn ngữ độ chính xác cao

---

## 🤝 Đóng góp

1. Fork và clone
2. Tạo nhánh: `git checkout -b feat/your-idea`
3. Commit và push
4. Mở PR

Với thay đổi lớn, vui lòng kèm theo:

- Mô tả ngắn về thay đổi hành vi mong đợi
- Ví dụ lệnh có thể tái hiện
- Đoạn phụ đề trước/sau khi phù hợp

---

## 📄 Giấy phép

MIT © Lachlan Chen
