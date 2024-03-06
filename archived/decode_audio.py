import torch
import whisper
import numpy as np
from datetime import timedelta
from pprint import pprint

from vad import vad_segments  # Ensure vad.py is properly structured for import

def get_speech_timestamps_from_vad(audio_path, target_sample_rate=16000):
    # Use vad_segments function from vad.py to get speech segments
    segments = vad_segments(audio_path, target_sample_rate=target_sample_rate)
    
    # Convert segments format if necessary to match the expected format
    speech_timestamps = [{'start': start, 'end': end} for start, end in segments]
    
    return speech_timestamps


# Function to format timestamps for subtitles
def format_timestamp(seconds):
    return str(timedelta(seconds=seconds))

def adjust_segment_duration(speech_timestamps, audio_length, target_duration_sec=30, sampling_rate=16000):
    """
    Independently adjust each speech segment to try to reach a minimum duration of 30 seconds
    by extending its end forward first, and if that's not sufficient, by extending its start backward
    without surpassing the 30-second threshold.
    """
    adjusted_segments = []

    for i, segment in enumerate(speech_timestamps):
        start, end = segment['start'], segment['end']
        current_duration = (end - start) / sampling_rate

        # Extend end forward if possible
        while current_duration < target_duration_sec and i < len(speech_timestamps) - 1:
            next_segment_start = speech_timestamps[i + 1]['start']
            possible_end_extension = next_segment_start
            if (possible_end_extension - start) / sampling_rate >= target_duration_sec:
                end = possible_end_extension
                break
            current_duration = (possible_end_extension - start) / sampling_rate
            i += 1

        # If still not enough, extend start backward
        if current_duration < target_duration_sec and i > 0:
            last_valid_start = start  # Track the last valid start position before surpassing 30s
            for j in range(i - 1, -1, -1):
                previous_segment_end = speech_timestamps[j]['end']
                possible_start_extension = previous_segment_end
                extended_duration = (end - possible_start_extension) / sampling_rate
                if extended_duration >= target_duration_sec:
                    start = last_valid_start  # Use the last valid start position that didn't surpass 30s
                    break
                else:
                    last_valid_start = possible_start_extension  # Update last valid start as we haven't surpassed 30s yet
                current_duration = extended_duration

        adjusted_segments.append({'start': start, 'end': end})

    return adjusted_segments


# Load Whisper model
whisper_model = whisper.load_model("large")

# Load VAD model
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, _, read_audio, *_) = vad_utils

# Load and process audio at 16 kHz suitable for VAD
audio_path = 'IMG_6276.wav'
sampling_rate_low = 16000
# sampling_rate_high = 44100
sampling_rate_high = 16000
audio = read_audio(audio_path, sampling_rate=sampling_rate_low)  # This audio is now at 16 kHz
audio_44100 = read_audio(audio_path, sampling_rate=sampling_rate_high)
# audio = read_audio(audio_path)  # This audio is now at 16 kHz
# Load audio and pad/trim it to fit 30 seconds
# audio_44100 = whisper.load_audio(audio_path)
# audio_44100 = whisper.pad_or_trim(audio)

# Detect speech segments using VAD

speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=sampling_rate_low)
# Instead of using get_speech_timestamps from Silero VAD
# speech_timestamps = get_speech_timestamps_from_vad(audio_path, target_sample_rate=16000)


pprint(speech_timestamps)

print("length of audio: ", len(audio))
print("length of audio_44100: ", len(audio_44100))

print(f"Detected {len(speech_timestamps)} speech segments")

speech_timestamps.insert(0, {'end': speech_timestamps[0]["start"], 'start': 0})

for i, _ in enumerate(speech_timestamps):
    if i == len(speech_timestamps) - 1:
        continue
    if i == 0:
        continue

    speech_timestamps[i]["end"] = speech_timestamps[i+1]["start"]
    speech_timestamps[i]["start"] = speech_timestamps[i-1]["end"]

# speech_timestamps = adjust_segment_duration(speech_timestamps, len(audio), sampling_rate=16000)


pprint(speech_timestamps)

# Process each speech segment detected by VAD
for i, segment in enumerate(speech_timestamps, start=1):
    start_frame, end_frame = segment['start'], segment['end']
    # Convert milliseconds to samples
    start_sample = int(start_frame * (sampling_rate_high / sampling_rate_low))
    end_sample = int(end_frame * (sampling_rate_high / sampling_rate_low))
    
    # Extract audio segment
    segment_audio = audio_44100[start_sample:end_sample]
    
    # Pad or trim the audio segment to fit into Whisper's expected input length
    segment_audio = whisper.pad_or_trim(segment_audio)
    
    # Convert the segment to Mel spectrogram for Whisper decoding
    mel = whisper.log_mel_spectrogram(segment_audio, n_mels=128).to(whisper_model.device)
    
    # Detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode the audio segment
    options = whisper.DecodingOptions()
    result = whisper_model.decode(mel, options)
    
    # Format and print the result like subtitles
    start_time = format_timestamp(start_frame / sampling_rate_low)
    end_time = format_timestamp(end_frame / sampling_rate_low)
    print(f"{i}\n{start_time} --> {end_time}\n{result.text}\n")
