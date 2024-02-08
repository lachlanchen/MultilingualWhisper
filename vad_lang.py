import torch
from pprint import pprint
import whisper

# Set the number of threads for PyTorch
torch.set_num_threads(1)

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps, _, read_audio, *_) = utils

# Specify the sampling rate
sampling_rate = 16000  # Silero VAD supports 8000 or 16000 Hz

# Load your audio file
wav = read_audio('IMG_6276.wav', sampling_rate=sampling_rate)

# Get speech timestamps from the audio file using Silero VAD
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)

# Load Whisper model for language detection
whisper_model = whisper.load_model("large-v2")

def predict_language_for_segment(audio_segment):
    # Make log-Mel spectrogram with 80 Mel bands (Whisper default)
    mel = whisper.log_mel_spectrogram(audio=audio_segment, n_mels=80).to(whisper_model.device)
    
    # Detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    return max(probs, key=probs.get)

# Process each speech segment to predict language
for segment in speech_timestamps:
    start_ms, end_ms = segment['start'], segment['end']
    # Convert milliseconds to samples
    start_sample = int(start_ms * sampling_rate / 1000)
    end_sample = int(end_ms * sampling_rate / 1000)
    
    # Extract audio segment
    audio_segment = wav[start_sample:end_sample]
    
    # Pad or trim the audio segment to fit into Whisper's expected input length if necessary
    audio_segment = whisper.pad_or_trim(audio_segment)
    
    # Predict language for the segment
    detected_language = predict_language_for_segment(audio_segment)
    print(f"Segment {segment}: Detected language: {detected_language}")
