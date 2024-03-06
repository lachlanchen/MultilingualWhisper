import whisper

# Load the model
model = whisper.load_model("large")

# Load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("IMG_6276.wav")
audio = whisper.pad_or_trim(audio)

# Make log-Mel spectrogram with 128 Mel bands and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio
options = whisper.DecodingOptions()
result = model.decode(mel, options)

# Print the recognized text
print(result.text)
