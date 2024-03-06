import torch
import whisper
from datetime import timedelta
from pprint import pprint

# Load Whisper model
whisper_model = whisper.load_model("large-v2")

# Load VAD model and utilities from Silero
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(_, _, read_audio, *_) = vad_utils

# Function to format timestamps for subtitles
def format_timestamp(seconds):
    return str(timedelta(seconds=seconds))

# Function to adjust speech segments to ensure a minimum duration of 30 seconds
def adjust_segments(segments, audio_length, target_duration=30):
    adjusted_segments = []
    for segment in segments:
        start, end = segment['start'], segment['end']
        duration = end - start
        if duration < target_duration:
            # Try extending forward
            potential_end = min(end + (target_duration - duration), audio_length)
            if potential_end - start < target_duration:
                # If not enough, extend backward
                start = max(0, start - (target_duration - (potential_end - start)))
            else:
                end = potential_end
        adjusted_segments.append({'start': start, 'end': end})
    return adjusted_segments

# Load and process audio
audio_path = 'IMG_6276.wav'
audio = read_audio(audio_path, sampling_rate=16000)

# Detect speech segments using VAD
# Uncomment the relevant block depending on the VAD method you wish to use

# Using Silero VAD
speech_timestamps = []  # Placeholder for speech timestamps from Silero VAD
speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=16000)

# Using custom VAD (get_speech_timestamps_from_vad function)
# Ensure vad.py is properly structured for import
# from vad import vad_segments
# speech_timestamps = [{'start': start, 'end': end} for start, end in vad_segments(audio_path, target_sample_rate=16000)]

print("Length of audio:", len(audio))
print(f"Detected {len(speech_timestamps)} speech segments")

# Adjust segments to ensure a minimum of 30 seconds
audio_length_seconds = len(audio) / 16000  # Assuming audio is sampled at 16kHz
adjusted_segments = adjust_segments(speech_timestamps, audio_length_seconds)

# Process each adjusted speech segment with Whisper
for i, segment in enumerate(adjusted_segments, start=1):
    start_frame, end_frame = segment['start'], segment['end']
    # Extract and process audio segment for Whisper
    segment_audio = whisper.pad_or_trim(audio[start_frame:end_frame])
    mel = whisper.log_mel_spectrogram(segment_audio).to(whisper_model.device)
    # Transcribe audio segment
    result = whisper_model.decode(mel, whisper.DecodingOptions()).text
    # Format and print the result like subtitles
    start_time = format_timestamp(start_frame)
    end_time = format_timestamp(end_frame)
    print(f"{i}\n{start_time} --> {end_time}\n{result}\n")
