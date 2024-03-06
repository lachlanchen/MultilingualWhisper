import torch
import whisper
import torchaudio
from io import BytesIO
import tempfile
import os
from pprint import pprint
import json
import subprocess
import argparse
from tqdm import tqdm
import subprocess
import shutil
import os
import tempfile

# import whisper_timestamped as whisper

from lingua import Language, LanguageDetectorBuilder

import traceback



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
        return 'und'

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


# def clean_timestamps(speech_timestamps, audio_length):
#     """
#     Adjusts the start and end frames of speech segments to ensure consitency.
    
#     :param speech_timestamps: List of dictionaries with 'start' and 'end' keys for each speech segment.
#     :param audio_length: The total length of the audio. 
#     """
#     num_segments = len(speech_timestamps)

#     if num_segments == 0:
#         return  # No segments to adjust

#     if num_segments == 1:
#         return 
#     else:
#         for i in range(num_segments):
#             if i == 0:
#                 speech_timestamps[i]["start"] = max(speech_timestamps[i]["start"], 0)
#                 speech_timestamps[i]["end"] = min(min(speech_timestamps[i]["end"], speech_timestamps[i+1]["start"]), audio_length)
#             elif i == num_segments - 1:
               
#                 speech_timestamps[i]["start"] = max(max(speech_timestamps[i-1]["end"], speech_timestamps[i]["start"]), 0)
#                 speech_timestamps[i]["end"] = min(speech_timestamps[i]["end"], audio_length)
#             else:
#                 # For middle segments, ensure no gaps by adjusting starts and ends to match neighbors
#                 speech_timestamps[i]["start"] = max(max(speech_timestamps[i-1]["end"], speech_timestamps[i]["start"]), 0)
#                 speech_timestamps[i]["end"] = min(min(speech_timestamps[i]["end"], speech_timestamps[i+1]["start"]), audio_length)

def clean_timestamps(speech_timestamps, audio_length):
    num_segments = len(speech_timestamps)

    if num_segments == 0:
        return  # No segments to adjust

    delete_mask = []
    for i in range(num_segments):
        if speech_timestamps[i]["lang"] == "und" or speech_timestamps[i]["text"] == []:
            delete_mask.append(i)

        if i == 0:
            # Ensure the first segment starts within the audio bounds
            speech_timestamps[i]["start"] = max(speech_timestamps[i]["start"], 0)
        else:
            # Ensure the start does not precede the previous segment's end
            speech_timestamps[i]["start"] = max(speech_timestamps[i-1]["end"], speech_timestamps[i]["start"])

        if i < num_segments - 1:
            # Adjust the end to not overlap with the next segment
            speech_timestamps[i]["end"] = min(speech_timestamps[i]["end"], speech_timestamps[i+1]["start"])
        else:
            # Ensure the last segment ends within the audio bounds
            speech_timestamps[i]["end"] = min(speech_timestamps[i]["end"], audio_length)

    for i in delete_mask[::-1]:
        speech_timestamps.remove(speech_timestamps[i])



def predict_language_for_segment(audio_segment, allowed_languages=["en", "zh", "ja", "ar", "yue"], model="large-v2"):
    # Convert audio segment to Mel spectrogram

    audio_segment = whisper.pad_or_trim(audio_segment)

    if model in ["large-v2"]:
        mel = whisper.log_mel_spectrogram(audio=audio_segment).to(whisper_model.device)
    else:
        mel = whisper.log_mel_spectrogram(audio=audio_segment, n_mels=128).to(whisper_model.device)

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
        segments = result["segments"]
        # pprint(result)
    except Exception as e:
        print(f"Error transcribing segment: {e}")
        traceback.print_exc()
        transcription = ""
        segments = []

        raise

    # Clean up the temporary file
    os.remove(temp_file.name)
    return transcription, segments, detected_language

def update_segments_with_language(words_segments, parent_start_frame, sampling_rate, detector, detected_language=None):
    new_segments = []

    for segment in words_segments:
        # segment_text = ' '.join([word['word'] for word in segment['words']])
        segment_text = segment["text"]
        text_language = detect_language_with_lingua(segment_text, detector)  # Use the Lingua function for language detection
        
        # Optimize the logic for determining segment language
        segment_language = text_language if detected_language != "yue" or text_language != "zh" else "yue"


        # Adjust start and end frames relative to the parent segment's start frame
        start_frame = int(segment['start'] * sampling_rate) + parent_start_frame
        end_frame = int(segment['end'] * sampling_rate) + parent_start_frame
        
        new_segments.append({
            'start': start_frame,
            'end': end_frame,
            'lang': segment_language,
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

    print("srt path: ", srt_path)
    print("json path: ", json_path)
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_content)



def process_audio_segments(speech_timestamps, wav, sampling_rate, model_name, whisper_model, detector):
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

    print("Transcribing audio...")

    # init run to obtain the language of each segment
    transcription_timestamps_with_lang = []

    for segment in tqdm(speech_timestamps):
        start_frame, end_frame = segment['start'], segment['end']
        start_sample, end_sample = int(start_frame), int(end_frame)
        # segment_audio = whisper.pad_or_trim(wav[start_sample:end_sample])
        segment_audio = wav[start_sample:end_sample]
        
        detected_language = predict_language_for_segment(segment_audio, model=model_name)  # Initial language detection
        transcription, words_segments, _ = transcribe_segment(segment_audio, start_frame, end_frame, sampling_rate, detected_language)
        
        # Update segments with detailed language detection and adjust frames
        new_segments = update_segments_with_language(
            words_segments, 
            start_frame, 
            sampling_rate, 
            detector,
            detected_language=detected_language
        )
        transcription_timestamps_with_lang.extend(new_segments)
        print(f"Segment {start_frame/sampling_rate}-{end_frame/sampling_rate}s (Language: {detected_language}): {transcription}")

    return transcription_timestamps_with_lang


def merge_segments(transcription_timestamps, sample_rate):
    print("Merging segments...")


    # merged based on the language detection from the result of whisper and langua
    merged_segments = []
    current_segment = None
    current_duration = 0



    for segment in tqdm(transcription_timestamps_with_lang):
        start_frame, end_frame = segment['start'], segment['end']
        segment_duration = (end_frame - start_frame) / sampling_rate
        
        if current_segment is None:
            # Initialize the first segment
            current_segment = segment
            current_duration = segment_duration
        # elif (current_segment['lang'] == segment['lang']) and (current_duration + segment_duration <= 30):
        elif (current_segment['lang'] == segment['lang']):
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



# def extract_audio_from_video(video_path, output_audio_path):
#     """
#     Extracts the audio from a video file and saves it as a WAV file.

#     Parameters:
#     - video_path: Path to the input video file.
#     - output_audio_path: Path where the extracted audio WAV file should be saved.
#     """
#     # Construct the ffmpeg command to extract audio
#     command = [
#         'ffmpeg', 
#         '-y',  # Overwrite output file if it exists
#         '-i', video_path,  # Input video file
#         '-vn',  # No video
#         '-acodec', 'pcm_s16le',  # WAV audio codec
#         '-ar', '44100',  # Set audio sampling rate to 16000 Hz
#         '-ac', '1',  # Set audio channels to mono
#         output_audio_path  # Output audio file
#     ]

#     # Execute the command
#     subprocess.run(command, check=True)
#     # Execute the command with stdout and stderr muted
#     # subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# def extract_audio_from_video(video_path, output_audio_path):
#     """
#     Tries to extract audio from a video file. If an exception occurs (likely due to file corruption),
#     it repairs the video outside the monitored folder and retries the extraction process.
    
#     Parameters:
#     - video_path: Path to the input video file.
#     - output_audio_path: Path where the extracted audio WAV file should be saved.
#     """
#     def run_ffmpeg_extract_audio(video_path, output_audio_path):
#         command = [
#             'ffmpeg',
#             '-y',  # Overwrite output file if it exists
#             '-i', video_path,  # Input video file
#             '-vn',  # No video
#             '-acodec', 'pcm_s16le',  # WAV audio codec
#             '-ar', '44100',  # Set audio sampling rate to 44100 Hz
#             '-ac', '1',  # Set audio channels to mono
#             output_audio_path  # Output audio file
#         ]
#         subprocess.run(command, check=True)

#     def repair_video(video_path):
#         # Use a temporary directory to repair the video
#         with tempfile.TemporaryDirectory() as temp_dir:
#             temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
#             # Attempt to repair the video by re-muxing to a temporary file
#             command = [
#                 'ffmpeg',
#                 '-y',  # Overwrite output file if it exists
#                 '-i', video_path,  # Input video file
#                 '-c', 'copy',  # Copy all streams without re-encoding
#                 '-movflags', '+faststart',  # Move moov atom to the beginning
#                 temp_video_path  # Temporary output video file
#             ]
#             subprocess.run(command, check=True)
#             # Replace the original file with the repaired file
#             shutil.move(temp_video_path, video_path)

#     try:
#         # First attempt to extract audio normally
#         run_ffmpeg_extract_audio(video_path, output_audio_path)
#     except subprocess.CalledProcessError:
#         print(f"Initial audio extraction failed, attempting to repair: {video_path}")
#         # If an exception occurs, repair the video outside the monitored folder
#         repair_video(video_path)
#         # Retry the audio extraction with the repaired video file
#         print("Retrying audio extraction after repair.")
#         run_ffmpeg_extract_audio(video_path, output_audio_path)
#         print(f"Audio extraction successful after repair: {output_audio_path}")


def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from a video file, enhances the audio for better transcription by increasing its volume,
    and saves it to a specified output path in WAV format. If an exception occurs, likely due to file corruption,
    it repairs the video outside the monitored folder and retries the extraction process.

    Parameters:
    - video_path: Path to the input video file.
    - output_audio_path: Path where the extracted audio WAV file should be saved.
    """

    def run_ffmpeg_extract_audio(video_path, output_audio_path):
        # Command to extract and enhance audio
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,  # Input video file
            '-vn',  # No video
            '-af', 'dynaudnorm=f=100',  # Dynamic audio normalization
            '-acodec', 'pcm_s16le',  # WAV audio codec
            '-ar', '44100',  # Set audio sampling rate to 44100 Hz
            '-ac', '1',  # Set audio channels to mono
            output_audio_path  # Output audio file
        ]
        subprocess.run(command, check=True)

    def repair_video(video_path):
        # Use a temporary directory to repair the video
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
            # Attempt to repair the video by re-muxing to a temporary file
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', video_path,  # Input video file
                '-c', 'copy',  # Copy all streams without re-encoding
                '-movflags', '+faststart',  # Move moov atom to the beginning
                temp_video_path  # Temporary output video file
            ]
            subprocess.run(command, check=True)
            # Replace the original file with the repaired file
            shutil.move(temp_video_path, video_path)

    try:
        # First attempt to extract and enhance audio normally
        run_ffmpeg_extract_audio(video_path, output_audio_path)
    except subprocess.CalledProcessError:
        print(f"Initial audio extraction failed, attempting to repair: {video_path}")
        # If an exception occurs, repair the video outside the monitored folder
        repair_video(video_path)
        # Retry the audio extraction with the repaired video file
        print("Retrying audio extraction after repair.")
        run_ffmpeg_extract_audio(video_path, output_audio_path)
        print(f"Audio extraction successful after repair: {output_audio_path}")

def clean_subtitles_dict(subtitles):
    """
    Cleans up subtitles by removing specific unwanted lines in Chinese or English.

    :param subtitles: List of subtitle dictionaries with keys 'start', 'end', 'lang', 'text'
    :return: A cleaned list of subtitle dictionaries
    """
    # Lines to be removed if detected
    unwanted_lines = [
        "优优独播",
        "优优独播剧场",
        "YoYo Television",
        "YoYo Television Series Exclusive",
    ]

    # Function to check if the subtitle text matches any unwanted lines
    def is_unwanted_line(text):
        for line in unwanted_lines:
            if line in text:
                return True
        return False

    # Filter out subtitles with unwanted text
    cleaned_subtitles = [subtitle for subtitle in subtitles if not is_unwanted_line(subtitle['text'])]

    return cleaned_subtitles

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Generate subtitles from a video file.")
    # parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument('-t', '--video-path', required=True, help="Path to the video file.")
    parser.add_argument("--whisper-model", default="large", help="Whisper model to use (default: large).")
    parser.add_argument("--force", action='store_true', help="Force overwrite of existing audio and subtitle files.")

    args = parser.parse_args()

    video_path = args.video_path
    model_name = args.whisper_model
    force = args.force

    # Determine the output audio path based on the input video path
    base_path, _ = os.path.splitext(video_path)
    audio_path = f"{base_path}.wav"



    # Change the SRT and JSON path based on the input audio/video path
    base_path, _ = os.path.splitext(audio_path)
    srt_path = f"{base_path}.srt"
    json_path = f"{base_path}.json"


    if not (os.path.exists(srt_path) or os.path.exists(json_path)) or args.force:
        

        # Initialize the Lingua language detector with specified languages
        languages = [Language.ENGLISH, Language.CHINESE, Language.JAPANESE, Language.ARABIC]  # Adjust languages as needed
        detector = LanguageDetectorBuilder.from_languages(*languages)\
            .with_minimum_relative_distance(0.9)\
            .build()


        # Set the number of threads for PyTorch
        torch.set_num_threads(1)

        # Load the Silero VAD model
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, _, read_audio, *_) = utils
        # Load Whisper model for language detection and transcription
        whisper_model = whisper.load_model(model_name)

        # Specify the sampling rate
        sampling_rate = 16000  # Hz

        # audio_path = '/home/lachlan/Projects/whisper_with_lang_detect/IMG_6276.wav'
        # video_path = '/home/lachlan/Projects/whisper_with_lang_detect/IMG_6276.MOV'

        
        # Extract audio from the video
        extract_audio_from_video(video_path, audio_path)
        
        

        # Load your audio file
        wav = read_audio(audio_path, sampling_rate=sampling_rate)

        # Get speech timestamps from the audio file using Silero VAD
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
        audio_length = len(wav)

        adjust_timestamps(speech_timestamps, audio_length)

        # init run to obtain the language of each segment
        transcription_timestamps_with_lang = process_audio_segments(speech_timestamps, wav, sampling_rate, model_name, whisper_model, detector)
        clean_timestamps(transcription_timestamps_with_lang, audio_length)

        # merged based on the language detection from the result of whisper and langua
        merged_segments = merge_segments(transcription_timestamps_with_lang, sampling_rate)

        # print("Transcribe merged VAD...")
        adjust_timestamps(merged_segments, audio_length)


        # second transcription of the language-specified and merged segments
        final_subtitles = process_audio_segments(merged_segments, wav, sampling_rate, model_name, whisper_model, detector)
        clean_timestamps(final_subtitles, audio_length)

        print("Final subtitles: ")
        for line in final_subtitles:
            print(line)

        # convert the final subtitles into a real subtitles

        # srt_path = "subtitles.srt"  # Specify the save path for the SRT file
        # json_path = "subtitles.json"  # Specify the save path for the JSON file

        

        # Generate SRT and JSON content
        srt_content = subtitles_to_srt(final_subtitles, sampling_rate)
        json_content = subtitles_to_json(final_subtitles, sampling_rate)

        # Save the subtitles
        save_subtitles(srt_content, json_content, srt_path, json_path)


        

        # generator = SubtitleGenerator(whisper_model=args.whisper_model, force=args.force)
        # generator.process_video(args.video_path)
