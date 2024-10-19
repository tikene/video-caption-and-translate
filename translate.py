import argparse
import re
import os
from openai import OpenAI
from faster_whisper import WhisperModel
import yt_dlp
import unicodedata
import ffmpeg



def sanitize_filename(filename):
    # Remove non-ASCII characters
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove any other invalid characters
    filename = re.sub(r'[^\w\-_\. ]', '', filename)
    return filename

def get_video_title(input_path):
    if input_path.startswith(('http://', 'https://', 'www.')):
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(input_path, download=False)
            return sanitize_filename(info['title'])
    else:
        return os.path.splitext(os.path.basename(input_path))[0]


def download_video(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': {
            'default': os.path.join(output_path, '%(title)s.%(ext)s')
        },
        'postprocessors': [{
            'key': 'FFmpegMetadata',
            'add_metadata': True,
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        sanitized_title = sanitize_filename(info['title'])
        ydl_opts['outtmpl']['default'] = os.path.join(output_path, f'{sanitized_title}.%(ext)s')
        
        ydl.params.update(ydl_opts)
        ydl.download([url])
        
        filename = os.path.join(output_path, f'{sanitized_title}.mp4')
    
    return filename


def transcribe_audio(audio_path, cache_path, model_size="large-v3"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=cache_path)
    
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability}")
    
    transcript = ""
    for i, segment in enumerate(segments, start=1):
        start_time = f"{int(segment.start // 3600):02d}:{int((segment.start % 3600) // 60):02d}:{segment.start % 60:06.3f}".replace(".", ",")
        end_time = f"{int(segment.end // 3600):02d}:{int((segment.end % 3600) // 60):02d}:{segment.end % 60:06.3f}".replace(".", ",")
        transcript += f"{i}\n{start_time} --> {end_time}\n{segment.text.lstrip()}\n\n"
    
    return transcript

def transcribe_audio_chatgpt(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="verbose_json"
        )
    
    transcript = ""
    for i, segment in enumerate(response.segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        transcript += f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
    
    print(f"Detected language '{response.language}'")
    
    return transcript

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

def extract_segments(transcript):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\d+\n\d{2}:\d{2}:\d{2},\d{3}).)+)'
    matches = re.findall(pattern, transcript, re.DOTALL)
    
    segments = []
    for match in matches:
        number, start, end, text = match
        segments.append({
            'number': int(number),
            'start': start.strip(),
            'end': end.strip(),
            'text': text.strip()
        })
    
    return segments

def translate_bulk(client, segments, target_language):
    text_to_translate = "\n\n".join([f"[SEG{s['number']}]\n{s['text']}" for s in segments])
    
    messages=[
        {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Maintain the [SEG#] markers and structure. Each [SEG#] is part of the same video. Return only the translated text with [SEG#] markers."},
        {"role": "user", "content": text_to_translate}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    translated_text = response.choices[0].message.content.strip()
    
    translated_segments = re.split(r'\[SEG\d+\]\s*', translated_text)
    translated_segments = [seg.strip() for seg in translated_segments if seg.strip()]
    
    return translated_segments

def process_transcript(transcript, target_language, api_key):
    client = OpenAI(api_key=api_key)
    if not client.api_key:
        raise ValueError("Invalid OpenAI API key.")

    segments = extract_segments(transcript)
    translated_segments = translate_bulk(client, segments, target_language)

    print(f"Number of original segments: {len(segments)}")
    print(f"Number of translated segments: {len(translated_segments)}")
    
    if len(segments) != len(translated_segments):
        print(f"ERROR: Original and translated segments don't match! Check the transcription for issues")

    translated_srt = ""
    translated_segments_with_timing = []
    for i, original in enumerate(segments):
        translated_srt += f"{original['number']}\n"
        translated_srt += f"{original['start']} --> {original['end']}\n"
        if i < len(translated_segments):
            translated_text = translated_segments[i].strip()
            translated_srt += f"{translated_text}\n\n"
            translated_segments_with_timing.append({
                'number': original['number'],
                'start': original['start'],
                'end': original['end'],
                'text': translated_text
            })
        else:
            translated_srt += "Translation not available\n\n"

    return translated_srt, translated_segments_with_timing

def save_translated_srt(translated_srt, target_language, output_dir, filename='transcript_translated.srt'):
    base_name = os.path.splitext(filename)[0]
    new_filename = f"{base_name}_{target_language.lower()}.srt"
    full_path = os.path.join(output_dir, new_filename)
    with open(full_path, 'w', encoding='utf-8') as file:
        file.write(translated_srt)
    print(f"Translated subtitles saved to: {full_path}")
    return full_path

def get_video_dimensions(video_path):
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def calculate_font_size(video_width):
    # Base font size for a 1920px wide video
    base_font_size = 24
    base_width = 1920
    
    # Calculate the scaling factor
    scale_factor = video_width / base_width
    
    # Calculate the new font size, with a minimum of 12 and maximum of 24
    font_size = max(12, min(24, int(base_font_size * scale_factor)))
    
    return font_size

def add_subtitles_with_ffmpeg(video_path, srt_path, output_path, font='NanumGothic'):
    print(f"Adding subtitles to video...")
    print(f"Video path: {video_path}")
    print(f"SRT path: {srt_path}")
    print(f"Output path: {output_path}")

    try:
        # Get video dimensions
        width, height = get_video_dimensions(video_path)
        
        # Calculate dynamic font size
        font_size = calculate_font_size(width)
        
        print(f"Video dimensions: {width}x{height}")
        print(f"Calculated font size: {font_size}")

        # Input video
        input_video = ffmpeg.input(video_path)

        # Add subtitles with dynamic font size
        video_with_subtitles = input_video.video.filter('subtitles', srt_path, 
            force_style=f'Fontname={font},FontSize={font_size},PrimaryColour=&HFFFFFF,OutlineColour=&H40000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=35')

        # Add original audio
        audio = input_video.audio

        # Output
        output = ffmpeg.output(video_with_subtitles, audio, output_path, acodec='copy')

        # Run FFmpeg
        ffmpeg.run(output, overwrite_output=True)

        print(f"Video with translated subtitles saved to: {output_path}")
    except ffmpeg.Error as e:
        print("FFmpeg Error:")
        print(e.stderr.decode())
        raise

def main():
    parser = argparse.ArgumentParser(description="Video Subtitle Translator")
    parser.add_argument("input", help="URL or path to the input video file")
    parser.add_argument("target_language", help="Target language for translation (Spanish, English, French...)")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    parser.add_argument("--models_path", default="Models", help="Path to store Whisper models")
    parser.add_argument("--openai_api_key", help="OpenAI API key")
    parser.add_argument("--font", default="NanumGothic", help="Font to use for subtitles")
    parser.add_argument("--use_local_whisper", action="store_true", help="Use local Whisper model for transcription instead of ChatGPT's Whisper")
    
    args = parser.parse_args()
    
    # Check for OpenAI api key
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please provide an OpenAI API key either as an argument or set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    # Create paths
    os.makedirs(args.models_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the video title (for URL) or base name (for local file)
    video_title = get_video_title(args.input)

    # Determine if input is a URL or local file
    if args.input.startswith(('http://', 'https://', 'www.')):
        print("Downloading video from URL...")
        video_path = download_video(args.input, args.output_dir)
    else:
        video_path = args.input

    # Transcribe audio from video
    print("Transcribing audio...")
    if args.use_local_whisper:
        transcript = transcribe_audio(video_path, args.models_path)
    else:
        transcript = transcribe_audio_chatgpt(client, video_path)
    print(f"Audio transcribed!\n")
    print(transcript)

    # Translate the transcript using AI and save it to a file
    print(f"Translating to {args.target_language}...")
    translated_srt, translated_segments = process_transcript(transcript, args.target_language, api_key)
    translated_srt_path = save_translated_srt(translated_srt, args.target_language, args.output_dir, f"{video_title}_transcript_{args.target_language.lower()}.srt")
    print(f"Translated to {args.target_language}!")
    
    # Add subtitles to video using FFmpeg
    output_video_path = os.path.join(args.output_dir, f"{video_title}_{args.target_language.lower()}.mp4")
    add_subtitles_with_ffmpeg(video_path, translated_srt_path, output_video_path, args.font)

if __name__ == "__main__":
    main()