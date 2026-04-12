# vad_lang_subtitle.py Pipeline Notes

This file explains the main control flow and why each step exists. It is meant to be read alongside `vad_lang_subtitle.py`.

## Inputs and outputs

Input:
- `--video-path` (or `-t`): path to a video file.
- `--whisper-model`: Whisper model name (tiny...large, large-v2, large-v3).
- `--force`: regenerate outputs even if they exist.

Output (same base name as input):
- `*.wav`: normalized mono audio extracted from the video.
- `*.srt`: subtitle file.
- `*.json`: rich subtitle payload with timestamps, language, and word-level data.

The script skips work if `*.srt` or `*.json` already exist unless `--force` is set.

## High-level pipeline

1. Parse CLI args and derive `audio_path`, `srt_path`, and `json_path`.
2. Build the Lingua language detector (currently EN/ZH/JA/AR).
3. Limit Torch CPU threads to 1 for stability (`torch.set_num_threads(1)`).
4. Load Silero VAD (`torch.hub.load("snakers4/silero-vad")`).
5. Load the Whisper model (`whisper.load_model(model_name)`).
6. Extract and normalize audio from the video with FFmpeg.
7. Read the audio at 16 kHz and run VAD to get speech timestamps.
8. First transcription pass: per-VAD segment language detection and transcription.
9. Clean/merge segments and align timestamps.
10. Second pass: re-transcribe merged segments for more coherent subtitles.
11. Reduce subtitle length with multi-step splitting rules.
12. Serialize outputs to SRT and JSON.

## Audio extraction and normalization

`extract_audio_from_video` does:
- `ffmpeg -af dynaudnorm=f=100` to normalize volume.
- Mono (`-ac 1`), 44.1 kHz (`-ar 44100`), PCM WAV (`pcm_s16le`).
- If extraction fails, it tries to repair the container by re-muxing and retries.

## Voice Activity Detection (VAD)

Silero VAD splits the audio into speech segments:
- `read_audio(...)` loads at 16 kHz.
- `get_speech_timestamps(...)` provides `start`/`end` sample indices.
- `adjust_timestamps(...)` stretches edges to avoid gaps at the start/end.

## Language detection and transcription

### Per-segment language detection

`predict_language_for_segment(...)`:
- Uses Whisper’s `detect_language` on the segment’s Mel spectrogram.
- Chooses the best language from `allowed_languages`:
  `en, zh, ja, ar, yue, ko, vi, es, fr`.

### Per-segment transcription

`transcribe_segment(...)`:
- Writes a temp WAV for the segment.
- Calls `whisper_model.transcribe(..., word_timestamps=True)`.
- Returns transcript text and Whisper word segments.

### Lingua refinement

`update_segments_with_language(...)` (and related helpers):
- Uses Lingua on segment text for extra language confidence.
- Adjusts word start/end to be relative to the original audio timeline.

## Merging and cleanup

Key cleanup steps:
- `clean_timestamps(...)`: removes empty/undetected segments and enforces monotonic timing.
- `merge_segments(...)`: merges adjacent segments with compatible language.
- `refine_transcription_segments_with_punctuation(...)` and VAD-aware refinements
  to reduce noisy splits.
- `adjust_timestamps(...)`: final alignment to avoid gaps.

## Subtitle length reduction

`reduce_subtitle_length(...)` applies multiple passes:
1. `refine_with_vad_pauses(...)`
2. `split_by_spaces_for_non_spaced_languages(...)`
3. `enhanced_punctuation_split(...)`
4. `split_long_segments_by_length(...)`

This produces shorter, easier-to-read lines without breaking word-level timing.

## Outputs

`subtitles_to_srt(...)` and `subtitles_to_json(...)` build the final files.

JSON includes:
- `start`, `end` in seconds
- `lang` (best-guess language)
- `text`
- word-level timestamps when available

## Performance and stability notes

- GPU is preferred for large models, but CPU fallback is possible.
- If you hit `SIGILL` or `SIGSEGV`, it usually indicates a binary mismatch
  (Torch/torchaudio built with CPU flags not supported by the host).
- Try smaller models (e.g. `large-v2` or `medium`) and ensure your Torch
  build matches your CUDA version.

## Common debugging checks

- Confirm FFmpeg is installed (`ffmpeg -version`).
- Run a tiny model on a short clip to isolate model issues:
  `--whisper-model tiny`.
- Verify Torch GPU visibility:
  `torch.cuda.is_available()` within the same environment.
