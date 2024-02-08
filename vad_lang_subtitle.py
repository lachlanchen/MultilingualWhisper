import torch
import whisper
import torchaudio
from io import BytesIO
import tempfile
import os
from pprint import pprint
import json

# import whisper_timestamped as whisper

from lingua import Language, LanguageDetectorBuilder



def detect_language_with_lingua(text, detector):
    """
    Detects the language of a given text using Lingua.
    Returns the ISO 639-1 code of the detected language if detection is confident; otherwise, returns None.
    """
    try:
        language = detector.detect_language_of(text)
        return language.iso_code_639_1.name.lower()  # Use .name to get the ISO code as a string
    except Exception as e:
        print(f"Language detection failed: {e}")
        return None

def adjust_timestamps(speech_timestamps, audio_length=None):
    """
    Adjusts the start and end frames of speech segments to ensure continuity and completeness.
    
    :param speech_timestamps: List of dictionaries with 'start' and 'end' keys for each speech segment.
    :param audio_length: Optional. The total length of the audio. If provided, ensures the last segment ends at this time.
    """
    num_segments = len(speech_timestamps)

    if num_segments == 0:
        return  # No segments to adjust

    if num_segments == 1:
        # If only one segment, it spans the entire audio
        speech_timestamps[0]["start"] = 0
        speech_timestamps[0]["end"] = audio_length if audio_length is not None else speech_timestamps[0]["end"]
    else:
        for i in range(num_segments):
            if i == 0:
                # Set the start of the first segment to 0
                speech_timestamps[i]["start"] = 0
                # Ensure the first segment ends where the second segment starts
                speech_timestamps[i]["end"] = speech_timestamps[i + 1]["start"]
            elif i == num_segments - 1:
                # For the last segment, adjust the start to match the penultimate segment's end
                # and the end to the audio length if specified, else keep as is
                speech_timestamps[i]["start"] = speech_timestamps[i - 1]["end"]
                speech_timestamps[i]["end"] = audio_length if audio_length is not None else speech_timestamps[i]["end"]
            else:
                # For middle segments, ensure no gaps by adjusting starts and ends to match neighbors
                speech_timestamps[i]["start"] = speech_timestamps[i - 1]["end"]
                speech_timestamps[i]["end"] = speech_timestamps[i + 1]["start"]


def predict_language_for_segment(audio_segment, allowed_languages=["en", "zh", "ja", "ar"][:2]):
    # Convert audio segment to Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio=audio_segment).to(whisper_model.device)
    # Detect the spoken language
    _, probs = whisper_model.detect_language(mel)

    # Filter the probabilities to only include allowed languages
    # Initialize a filtered_probs dictionary with allowed languages as keys and a default probability of 0
    filtered_probs = {lang: probs.get(lang, 0) for lang in allowed_languages}

    # pprint(filtered_probs)
    # Find the language with the highest probability among the allowed languages
    max_language = max(filtered_probs, key=filtered_probs.get)

    return max_language


def transcribe_segment(audio_segment, start_frame, end_frame, sampling_rate, detected_language):
    # Create a temporary file for the audio segment
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    torchaudio.save(temp_file.name, audio_segment.unsqueeze(0), sampling_rate)
    temp_file.close()  # Close the file so Whisper can read it
    
    # Transcribe the audio segment using Whisper
    try:
        result = whisper_model.transcribe(temp_file.name, language=detected_language, word_timestamps=True)
        transcription = result["text"]
        # pprint(result)
    except Exception as e:
        print(f"Error transcribing segment: {e}")
        transcription = ""

    # Clean up the temporary file
    os.remove(temp_file.name)
    return transcription, result["segments"], detected_language

def update_segments_with_language(words_segments, parent_start_frame, sampling_rate, detector):
    new_segments = []

    for segment in words_segments:
        # segment_text = ' '.join([word['word'] for word in segment['words']])
        segment_text = segment["text"]
        detected_language = detect_language_with_lingua(segment_text, detector)  # Use the Lingua function for language detection
        
        # Adjust start and end frames relative to the parent segment's start frame
        start_frame = int(segment['start'] * sampling_rate) + parent_start_frame
        end_frame = int(segment['end'] * sampling_rate) + parent_start_frame
        
        new_segments.append({
            'start': start_frame,
            'end': end_frame,
            'lang': detected_language,
            'text': segment_text
        })

    return new_segments

def frames_to_milliseconds(frames, sample_rate):
    """Converts frame numbers to milliseconds based on the given sample rate."""
    return (frames / sample_rate) * 1000

def format_timestamp(ms):
    """Converts milliseconds to 'hh:mm:ss,ms' format."""
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    milliseconds = int(ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def subtitles_to_srt(subtitles, sample_rate):
    """Converts subtitles to SRT format."""
    srt_content = ""
    for i, subtitle in enumerate(subtitles, start=1):
        start_ms = frames_to_milliseconds(subtitle['start'], sample_rate)
        end_ms = frames_to_milliseconds(subtitle['end'], sample_rate)
        start_srt = format_timestamp(start_ms)
        end_srt = format_timestamp(end_ms)
        srt_content += f"{i}\n{start_srt} --> {end_srt}\n{subtitle['text']}\n\n"
    return srt_content


def subtitles_to_json(subtitles, sample_rate):
    """Converts subtitles to a JSON string, formatting timestamps as 'hh:mm:ss,ms'."""
    adjusted_subtitles = []
    for subtitle in subtitles:
        adjusted_subtitle = subtitle.copy()
        # Format start and end times as 'hh:mm:ss,ms' for JSON representation
        adjusted_subtitle['start'] = format_timestamp(frames_to_milliseconds(subtitle['start'], sample_rate))
        adjusted_subtitle['end'] = format_timestamp(frames_to_milliseconds(subtitle['end'], sample_rate))
        adjusted_subtitles.append(adjusted_subtitle)
    return json.dumps(adjusted_subtitles, indent=4, ensure_ascii=False)


def save_subtitles(srt_content, json_content, srt_path, json_path):
    """Saves SRT and JSON content to specified paths."""
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_content)


class SubtitleGenerator:
    def __init__(self, whisper_model='large', force=False):
        self.force = force
        self.whisper_model = whisper.load_model(whisper_model)
        self.detector = self.init_language_detector()

    def init_language_detector(self):
        languages = [Language.ENGLISH, Language.CHINESE, Language.JAPANESE, Language.ARABIC]
        return LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.9).build()

    def extract_audio(self, video_path):
        audio_path = os.path.splitext(video_path)[0] + '.wav'
        if not os.path.exists(audio_path) or self.force:
            subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path], check=True)
        return audio_path

    def detect_and_transcribe(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        return self.whisper_model.transcribe(waveform)

    def save_subtitles(self, transcription, base_path):
        srt_path = f"{base_path}.srt"
        json_path = f"{base_path}.json"
        
        with open(srt_path, "w") as srt_file, open(json_path, "w") as json_file:
            json_file.write("[\n")
            for i, segment in enumerate(transcription['segments'], 1):
                start = segment['start']
                end = segment['end']
                text = segment['text'].replace('"', '\\"')
                srt_file.write(f"{i}\n{self.format_timestamp(start)} --> {self.format_timestamp(end)}\n{text}\n\n")
                json_entry = f'    {{"start": "{self.format_timestamp(start)}", "end": "{self.format_timestamp(end)}", "text": "{text}"}}'
                if i < len(transcription['segments']):
                    json_entry += ","
                json_file.write(json_entry + "\n")
            json_file.write("]")

    def format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')

    def process_video(self, video_path):
        audio_path = self.extract_audio(video_path)
        transcription = self.detect_and_transcribe(audio_path)
        base_path = os.path.splitext(video_path)[0]
        self.save_subtitles(transcription, base_path)


def process_audio_segments(speech_timestamps, wav, sampling_rate, whisper_model, detector):
    """
    Process audio segments to obtain language and transcription for each segment.

    Parameters:
    - speech_timestamps: List of dictionaries indicating speech segments with 'start' and 'end' frames.
    - wav: The waveform of the audio file.
    - sampling_rate: The sampling rate of the audio file.
    - whisper_model: The Whisper model loaded for transcription.
    - detector: The language detector for identifying the language of a segment.

    Returns:
    - List of dictionaries with transcription and language detection for each segment.
    """
    # init run to obtain the language of each segment
    transcription_timestamps_with_lang = []

    for segment in speech_timestamps:
        start_frame, end_frame = segment['start'], segment['end']
        start_sample, end_sample = int(start_frame), int(end_frame)
        segment_audio = whisper.pad_or_trim(wav[start_sample:end_sample])
        
        detected_language = predict_language_for_segment(segment_audio)  # Initial language detection
        transcription, words_segments, _ = transcribe_segment(segment_audio, start_frame, end_frame, sampling_rate, detected_language)
        
        # Update segments with detailed language detection and adjust frames
        new_segments = update_segments_with_language(words_segments, start_frame, sampling_rate, detector)
        transcription_timestamps_with_lang.extend(new_segments)
        print(f"Segment {start_frame/sampling_rate}-{end_frame/sampling_rate}s (Language: {detected_language}): {transcription}")

    return transcription_timestamps_with_lang


def merge_segments(transcription_timestamps, sample_rate):
    # merged based on the language detection from the result of whisper and langua
    merged_segments = []
    current_segment = None
    current_duration = 0

    for segment in transcription_timestamps_with_lang:
        start_frame, end_frame = segment['start'], segment['end']
        segment_duration = (end_frame - start_frame) / sampling_rate
        
        if current_segment is None:
            # Initialize the first segment
            current_segment = segment
            current_duration = segment_duration
        elif (current_segment['lang'] == segment['lang']) and (current_duration + segment_duration <= 30):
            # Merge segments: extend duration and update end frame
            current_segment['end'] = end_frame
            current_duration += segment_duration
        else:
            # Current segment does not match or exceeds 30 seconds, start a new segment
            merged_segments.append(current_segment)
            current_segment = segment
            current_duration = segment_duration

    # Don't forget to add the last segment
    if current_segment:
        merged_segments.append(current_segment)

    return merged_segments


if __name__ == "__main__": 

    # Initialize the Lingua language detector with specified languages
    languages = [Language.ENGLISH, Language.CHINESE, Language.JAPANESE, Language.ARABIC]  # Adjust languages as needed
    detector = LanguageDetectorBuilder.from_languages(*languages)\
        .with_minimum_relative_distance(0.9)\
        .build()


    # Set the number of threads for PyTorch
    torch.set_num_threads(1)

    # Load the Silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    (get_speech_timestamps, _, read_audio, *_) = utils
    # Load Whisper model for language detection and transcription
    whisper_model = whisper.load_model("large-v2")

    # Specify the sampling rate
    sampling_rate = 16000  # Hz

    audio_path = '/home/lachlan/Projects/whisper_with_lang_detect/IMG_6276.wav'
    

    # Load your audio file
    wav = read_audio(audio_path, sampling_rate=sampling_rate)

    # Get speech timestamps from the audio file using Silero VAD
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    audio_length = len(wav)


    adjust_timestamps(speech_timestamps, audio_length)


    # # init run to obtain the language of each segment
    # transcription_timestamps_with_lang = []

    # for segment in speech_timestamps:
    #     start_frame, end_frame = segment['start'], segment['end']
    #     start_sample, end_sample = int(start_frame), int(end_frame)
    #     segment_audio = whisper.pad_or_trim(wav[start_sample:end_sample])
        
    #     detected_language = predict_language_for_segment(segment_audio)  # Initial language detection
    #     transcription, words_segments, _ = transcribe_segment(segment_audio, start_frame, end_frame, sampling_rate, detected_language)
        
    #     # Update segments with detailed language detection and adjust frames
    #     new_segments = update_segments_with_language(words_segments, start_frame, sampling_rate, detector)
    #     transcription_timestamps_with_lang.extend(new_segments)
    #     print(f"Segment {start_frame/sampling_rate}-{end_frame/sampling_rate}s (Language: {detected_language}): {transcription}")

    transcription_timestamps_with_lang = process_audio_segments(speech_timestamps, wav, sampling_rate, whisper_model, detector)
    

    # # merged based on the language detection from the result of whisper and langua
    # merged_segments = []
    # current_segment = None
    # current_duration = 0

    # for segment in transcription_timestamps_with_lang:
    #     start_frame, end_frame = segment['start'], segment['end']
    #     segment_duration = (end_frame - start_frame) / sampling_rate
        
    #     if current_segment is None:
    #         # Initialize the first segment
    #         current_segment = segment
    #         current_duration = segment_duration
    #     elif (current_segment['lang'] == segment['lang']) and (current_duration + segment_duration <= 30):
    #         # Merge segments: extend duration and update end frame
    #         current_segment['end'] = end_frame
    #         current_duration += segment_duration
    #     else:
    #         # Current segment does not match or exceeds 30 seconds, start a new segment
    #         merged_segments.append(current_segment)
    #         current_segment = segment
    #         current_duration = segment_duration

    # # Don't forget to add the last segment
    # if current_segment:
    #     merged_segments.append(current_segment)

    merged_segments = merge_segments(transcription_timestamps_with_lang, sampling_rate)

    
    # second transcription of the language-specified and merged segments

    print("Transcribe merged VAD...")
    adjust_timestamps(merged_segments, audio_length)

    # final_subtitles = []

    # # Optionally, print or process merged segments
    # for segment in merged_segments:
    #     # print(f"Merged Segment {segment['start']/sampling_rate}-{segment['end']/sampling_rate}s (Language: {segment['lang']}): Duration {segment['end'] - segment['start']} samples")
    #     start_frame, end_frame = segment['start'], segment['end']
    #     start_sample = int(start_frame)
    #     end_sample = int(end_frame)
    #     segment_audio = wav[start_sample:end_sample]
    #     segment_audio = whisper.pad_or_trim(segment_audio)  # Adjust segment length if necessary
        
    #     detected_language = predict_language_for_segment(segment_audio)
    #     transcription, words_segments, _ = transcribe_segment(segment_audio, start_frame, end_frame, sampling_rate, detected_language)
    #     # Update segments with detailed language detection and adjust frames
    #     new_segments = update_segments_with_language(words_segments, start_frame, sampling_rate, detector)
    #     final_subtitles.extend(new_segments)
    #     print(f"Segment {start_frame/sampling_rate}-{end_frame/sampling_rate}s (Language: {detected_language}): {transcription}")

    final_subtitles = process_audio_segments(merged_segments, wav, sampling_rate, whisper_model, detector)


    for line in final_subtitles:
        print(line)

    # convert the final subtitles into a real subtitles

    srt_path = "subtitles.srt"  # Specify the save path for the SRT file
    json_path = "subtitles.json"  # Specify the save path for the JSON file

    # Generate SRT and JSON content
    srt_content = subtitles_to_srt(final_subtitles, sampling_rate)
    json_content = subtitles_to_json(final_subtitles, sampling_rate)

    # Save the subtitles
    save_subtitles(srt_content, json_content, srt_path, json_path)


    # parser = argparse.ArgumentParser(description="Generate subtitles from a video file.")
    # parser.add_argument("video_path", help="Path to the video file.")
    # parser.add_argument("--whisper-model", default="large", help="Whisper model to use (default: large).")
    # parser.add_argument("--force", action='store_true', help="Force overwrite of existing audio and subtitle files.")

    # args = parser.parse_args()

    # generator = SubtitleGenerator(whisper_model=args.whisper_model, force=args.force)
    # generator.process_video(args.video_path)
