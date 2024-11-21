import argparse
import re
import os
from openai import OpenAI
from faster_whisper import WhisperModel
import yt_dlp
import unicodedata
import ffmpeg
import math
from pydub import AudioSegment
import tempfile
import shutil 
import subprocess
import json

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

def split_audio(audio_path, max_size_mb=25):
    """Split audio file into chunks smaller than max_size_mb"""
    max_size_bytes = max_size_mb * 1024 * 1024
    print(f"Loading audio file: {audio_path}")
    
    try:
        audio = AudioSegment.from_mp3(audio_path)
        duration_ms = len(audio)
        
        # Calculate chunk size based on original file size and duration
        file_size = os.path.getsize(audio_path)
        ms_per_mb = duration_ms / (file_size / 1024 / 1024)
        chunk_duration_ms = int(ms_per_mb * (max_size_mb * 0.95))  # 5% safety margin
        
        chunks = []
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Split audio into chunks
        for i, start in enumerate(range(0, duration_ms, chunk_duration_ms)):
            end = min(start + chunk_duration_ms, duration_ms)
            chunk = audio[start:end]
            
            # Export chunk with optimal settings
            chunk_path = os.path.join(temp_dir, f'chunk_{i}.mp3')
            print(f"Exporting chunk {i+1} to: {chunk_path}")
            
            chunk.export(
                chunk_path,
                format="mp3",
                parameters=[
                    "-ac", "1",  # Mono audio
                    "-ar", "16000",  # 16kHz sample rate
                    "-q:a", "9"  # Lowest quality (highest compression)
                ]
            )
            
            # Verify chunk size
            chunk_size = os.path.getsize(chunk_path)
            if chunk_size > max_size_bytes:
                raise ValueError(
                    f"Chunk {i} size ({chunk_size/1024/1024:.2f}MB) exceeds limit "
                    f"({max_size_mb}MB) after compression"
                )
            
            chunks.append(chunk_path)
            print(f"Chunk {i+1} size: {chunk_size/1024/1024:.2f}MB")
        
        return chunks, temp_dir
        
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


def transcribe_audio_chatgpt(client, audio_path):
    """Modified transcription function to handle large files"""
    # First extract audio in optimal format
    audio_mp3_path = extract_audio(audio_path)
    temp_dir = None
    try:
        # Check if file is larger than 24MB
        if os.path.getsize(audio_mp3_path) > 1024 * 1024 * 24:
            print("Audio file larger than 25MB, splitting into chunks...")
            chunks, temp_dir = split_audio(audio_mp3_path)
            
            # Process each chunk
            all_segments = []
            for i, chunk_path in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                with open(chunk_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                    
                    # Adjust timestamps for chunks after the first
                    time_offset = i * (25 * 60)  # Approximate offset based on chunk duration
                    for segment in response.segments:
                        segment.start += time_offset
                        segment.end += time_offset
                        all_segments.append(segment)
        else:
            # Process single file if under 25MB
            with open(audio_mp3_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                all_segments = response.segments
        
        # Generate transcript with adjusted timestamps
        transcript = ""
        for i, segment in enumerate(all_segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            transcript += f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
        
        return transcript
            
    finally:
        # Clean up temporary files
        if os.path.exists(audio_mp3_path):
            os.remove(audio_mp3_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
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


def extract_audio(video_path, output_path=None):
    """Extract audio from video file using ffmpeg command line with proper encoding handling"""
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '.mp3'
    
    try:
        # Check if ffmpeg is installed
        if not shutil.which('ffmpeg'):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
        
        # Construct ffmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,  # Input file
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-ac', '1',  # Mono audio
            '-ar', '16000',  # 16kHz sampling rate
            '-y',  # Overwrite output file
            output_path
        ]
        
        # Run ffmpeg command with proper encoding handling
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=None if os.name != 'nt' else subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW)
        )
        
        # Read output using binary mode and decode manually
        stdout_data, stderr_data = process.communicate()
        
        # Check if the process was successful
        if process.returncode != 0:
            # Safely decode error output
            try:
                error_message = stderr_data.decode('utf-8', errors='replace')
            except:
                error_message = str(stderr_data)
            raise RuntimeError(f"Error extracting audio: {error_message}")
        
        return output_path
        
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    
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
    """Translate segments with enhanced natural language prompting"""
    text_to_translate = "\n\n".join([f"[SEG{s['number']}]\n{s['text']}" for s in segments])
    
    messages = [
        {
            "role": "system",
            "content": f"""You are an expert translator specializing in {target_language}, with deep understanding of cultural context and natural speech patterns. Your task is to translate the following video transcript segments.

Key translation principles to follow:
- Prioritize natural, conversational language over literal translations
- Maintain the original tone and style (casual, formal, humorous, etc.)
- Adapt idioms and expressions to culturally appropriate equivalents in {target_language}
- Ensure the translations sound fluid and native when spoken aloud
- Consider the context that this is spoken dialogue, not written text
- Preserve the emotional impact and intent of the original speech

Format requirements:
- Maintain the [SEG#] markers exactly as they appear
- Keep line breaks and spacing consistent
- Return only the translated text with segment markers, no explanations

Example of natural translation:
[SEG1] "Hey, what's up?" → [SEG1] "¿Qué tal?" (Spanish - casual greeting adapted to target culture)
Instead of: "¿Oye, qué está arriba?" (literal translation)"""
        },
        {
            "role": "user", 
            "content": text_to_translate
        }
    ]
    
    print(f"Requesting translation to {target_language}...")
    response = client.chat.completions.create(
        model="gpt-4",  # Use gpt-4o-mini to reduce api token costs, gpt-4 yields the best results at the highest cost
        messages=messages,
        temperature=0.7  # Slightly increased for more natural language
    )
    translated_text = response.choices[0].message.content.strip()
    
    # Process and validate the translated segments
    translated_segments = re.split(r'\[SEG\d+\]\s*', translated_text)
    translated_segments = [seg.strip() for seg in translated_segments if seg.strip()]
    
    # Validation check
    if len(translated_segments) != len(segments):
        print("Warning: Number of translated segments doesn't match original")
        
    return translated_segments

def process_transcript(transcript, target_language, api_key):
    """Enhanced transcript processing with better error handling"""
    client = OpenAI(api_key=api_key)
    if not client.api_key:
        raise ValueError("Invalid OpenAI API key.")

    # Extract segments first
    segments = extract_segments(transcript)
    
    # Split into smaller batches if needed (to avoid token limits)
    batch_size = 20  # Adjust based on typical segment length
    translated_segments = []
    
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        print(f"Translating batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}...")
        
        try:
            batch_translations = translate_bulk(client, batch, target_language)
            translated_segments.extend(batch_translations)
        except Exception as e:
            print(f"Error translating batch {i//batch_size + 1}: {str(e)}")
            # Add placeholder for failed translations
            translated_segments.extend(["[Translation error]"] * len(batch))

    print(f"Number of original segments: {len(segments)}")
    print(f"Number of translated segments: {len(translated_segments)}")
    
    if len(segments) != len(translated_segments):
        print("Warning: Translation mismatch. Some segments may be missing.")

    # Generate the final SRT content
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
    """Save translated SRT with proper path handling"""
    # Sanitize the base name to remove problematic characters
    base_name = os.path.splitext(filename)[0]
    base_name = sanitize_filename(base_name)
    
    # Create new filename with sanitized base name
    new_filename = f"{base_name}_{target_language.lower()}.srt"
    full_path = os.path.join(output_dir, new_filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the file with UTF-8 encoding
    with open(full_path, 'w', encoding='utf-8') as file:
        file.write(translated_srt)
    
    print(f"Translated subtitles saved to: {full_path}")
    return full_path

def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe with proper encoding handling"""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=None if os.name != 'nt' else subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW)
        )
        
        stdout_data, stderr_data = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Error getting video dimensions: {stderr_data.decode('utf-8', errors='replace')}")
        
        video_info = json.loads(stdout_data.decode('utf-8'))
        width = int(video_info['streams'][0]['width'])
        height = int(video_info['streams'][0]['height'])
        
        return width, height
        
    except Exception as e:
        print(f"Error getting video dimensions: {str(e)}")
        raise

def calculate_font_size(video_width):
    # Base font size for a 1920px wide video
    base_font_size = 16
    base_width = 1920
    
    # Calculate the scaling factor
    scale_factor = video_width / base_width
    
    # Calculate the new font size, with a minimum of 12 and maximum of 24
    font_size = max(12, min(24, int(base_font_size * scale_factor)))
    
    return font_size

def add_subtitles_with_ffmpeg(video_path, srt_path, output_path, font='NanumGothic'):
    """Add subtitles to video using ffmpeg with proper path handling"""
    print(f"Adding subtitles to video...")
    print(f"Video path: {video_path}")
    print(f"SRT path: {srt_path}")
    print(f"Output path: {output_path}")

    try:
        # Get video dimensions using ffprobe
        width, height = get_video_dimensions(video_path)
        font_size = calculate_font_size(width)
        
        print(f"Video dimensions: {width}x{height}")
        print(f"Calculated font size: {font_size}")
        
        # Normalize paths and escape special characters
        video_path = os.path.normpath(video_path)
        srt_path = os.path.normpath(srt_path)
        output_path = os.path.normpath(output_path)
        
        # On Windows, convert backslashes to forward slashes for ffmpeg
        if os.name == 'nt':
            video_path = video_path.replace('\\', '/')
            srt_path = srt_path.replace('\\', '/')
            output_path = output_path.replace('\\', '/')

        # Escape special characters in paths
        srt_path = srt_path.replace("'", "'\\''")
        
        # Construct subtitle filter with escaped paths
        subtitle_filter = f"subtitles='{srt_path}'"
        style = f":force_style='Fontname={font},FontSize={font_size},PrimaryColour=&HFFFFFF,OutlineColour=&H40000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=35'"
        
        # Construct complete ffmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', subtitle_filter + style,
            '-c:a', 'copy',
            '-y',
            output_path
        ]
        
        print("Executing command:", ' '.join(command))  # Debug print
        
        # Create startupinfo to hide console window on Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        # Run ffmpeg command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo
        )
        
        # Read output using binary mode and decode manually
        stdout_data, stderr_data = process.communicate()
        
        # Check if the process was successful
        if process.returncode != 0:
            error_message = stderr_data.decode('utf-8', errors='replace')
            if "No such file or directory" in error_message:
                print(f"Debug - Checking file existence:")
                print(f"Video exists: {os.path.exists(video_path)}")
                print(f"SRT exists: {os.path.exists(srt_path)}")
                print(f"Output directory exists: {os.path.exists(os.path.dirname(output_path))}")
            raise RuntimeError(f"Error adding subtitles: {error_message}")
            
        print(f"Video with translated subtitles saved to: {output_path}")
        
    except Exception as e:
        print(f"Error adding subtitles: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
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
        print("Using local whisper model")
        transcript = transcribe_audio(video_path, args.models_path)
    else:
        print("Using OpenAI API whisper model")
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