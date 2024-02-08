from pydub import AudioSegment
import webrtcvad
import os

def vad_segments(audio_file, target_sample_rate=16000, chunk_duration_ms=30):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)
    
    # Ensure audio is mono and resample to target sample rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sample_rate)
    
    # Convert audio to raw bytes
    audio_bytes = audio.raw_data
    sample_width = audio.sample_width
    num_channels = audio.channels
    
    # Initialize VAD
    vad = webrtcvad.Vad(1)  # Aggressiveness level
    
    # Calculate chunk size and number of chunks
    chunk_size = int(target_sample_rate * chunk_duration_ms / 1000) * sample_width * num_channels
    num_chunks = int(len(audio_bytes) / chunk_size)
    
    voice_segments = []
    
    # Process each chunk with VAD
    for i in range(num_chunks):
        chunk = audio_bytes[i*chunk_size:(i+1)*chunk_size]
        is_speech = vad.is_speech(chunk, target_sample_rate)
        if is_speech:
            start_ms = i * chunk_duration_ms
            end_ms = start_ms + chunk_duration_ms
            voice_segments.append((start_ms, end_ms))
    
    # Combine consecutive voice segments
    combined_segments = []
    for start, end in voice_segments:
        if not combined_segments or start > combined_segments[-1][1]:
            combined_segments.append([start, end])
        else:
            combined_segments[-1][1] = end

    return combined_segments

def save_segments(audio_file, segments, target_dir='segments'):
    audio = AudioSegment.from_file(audio_file)
    os.makedirs(target_dir, exist_ok=True)
    
    for i, (start_ms, end_ms) in enumerate(segments):
        segment_audio = audio[start_ms:end_ms]
        segment_file_name = os.path.join(target_dir, f'segment_{i+1}.wav')
        segment_audio.export(segment_file_name, format='wav')
        print(f'Saved: {segment_file_name}')

# Main execution
if __name__ == '__main__':
    audio_file = 'audio.wav'  # Ensure this path is correct
    segments = vad_segments(audio_file)
    print(segments)
    save_segments(audio_file, segments)

