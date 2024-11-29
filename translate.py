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
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from typing import List, Dict
import time
from tqdm import tqdm


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


def find_nearest_silence(audio: AudioSegment, position_ms: int, window_ms: int = 1000) -> int:
    """Find the nearest silence point to align chunk boundaries"""
    start = max(0, position_ms - window_ms)
    end = min(len(audio), position_ms + window_ms)
    
    # Extract the audio segment around the target position
    chunk = audio[start:end]
    
    # Calculate RMS values in small windows
    window_size = 50  # 50ms windows
    rms_values = []
    for i in range(0, len(chunk), window_size):
        window = chunk[i:i + window_size]
        rms_values.append(window.rms)
    
    # Find the quietest point
    if not rms_values:
        return position_ms
        
    min_rms_index = np.argmin(rms_values)
    silence_position = start + (min_rms_index * window_size)
    
    return silence_position


def process_chunk_segments(chunk_info: Dict, segments: List, video_duration: float) -> List:
    """Process segments from a chunk with precise timing"""
    from decimal import Decimal, ROUND_HALF_UP
    
    processed_segments = []
    base_time = Decimal(str(chunk_info['start_time']))
    has_previous = chunk_info['has_previous']
    has_next = chunk_info['has_next']
    overlap_start = Decimal(str(chunk_info['overlap_start']))
    overlap_end = Decimal(str(chunk_info['overlap_end']))
    
    for segment in segments:
        # Convert timestamps to Decimal for precise arithmetic
        start = Decimal(str(segment.start))
        end = Decimal(str(segment.end))
        
        # Adjust timestamps for chunk position
        actual_start = (start - (overlap_start - base_time)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        actual_end = (end - (overlap_start - base_time)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        
        # Skip segments in overlap regions except for chunk boundaries
        if has_previous and actual_start < base_time:
            continue
        if has_next and actual_end > Decimal(str(chunk_info['end_time'])):
            continue
            
        # Validate against video duration
        final_start = min(float(actual_start), video_duration)
        final_end = min(float(actual_end), video_duration)
        
        if final_start < final_end:
            segment.start = final_start
            segment.end = final_end
            processed_segments.append(segment)
    
    return processed_segments


def split_audio(audio_path: str, video_duration: float, max_size_mb: int = 25) -> tuple:
    """Split audio with improved chunk handling and overlap"""
    try:
        audio = AudioSegment.from_mp3(audio_path)
        duration_ms = len(audio)
        
        # Validate and adjust audio duration
        if abs(duration_ms/1000.0 - video_duration) > 1.0:
            print(f"Warning: Audio duration mismatch. Audio: {duration_ms/1000.0:.2f}s, Video: {video_duration:.2f}s")
            # Adjust audio duration if needed
            if duration_ms/1000.0 > video_duration:
                audio = audio[:int(video_duration * 1000)]
                duration_ms = len(audio)
        
        # Normalize audio
        audio = audio.normalize(headroom=0.1)
        
        # Calculate optimal chunk duration
        sample_width = audio.sample_width
        frame_rate = audio.frame_rate
        channels = audio.channels
        
        bytes_per_second = sample_width * frame_rate * channels
        optimal_chunk_duration_ms = int((max_size_mb * 1024 * 1024 * 0.90) / (bytes_per_second / 1000))
        
        # Add overlap between chunks
        overlap_duration = 2000  # 2 seconds overlap
        chunks = []
        temp_dir = tempfile.mkdtemp()
        
        current_position = 0
        chunk_number = 0
        
        while current_position < duration_ms:
            # Find optimal chunk boundary near silence
            target_end = min(current_position + optimal_chunk_duration_ms, duration_ms)
            end_position = find_nearest_silence(audio, target_end)
            
            # Ensure minimum chunk size
            if end_position - current_position < 1000:  # Minimum 1 second
                end_position = min(current_position + 1000, duration_ms)
            
            # Extract chunk with overlap
            chunk_start = max(0, current_position - overlap_duration)
            chunk_end = min(duration_ms, end_position + overlap_duration)
            chunk = audio[chunk_start:chunk_end]
            
            # Export chunk with quality settings
            chunk_path = os.path.join(temp_dir, f'chunk_{chunk_number}.mp3')
            print(f"Exporting chunk {chunk_number + 1} ({chunk_start/1000.0:.2f}s - {chunk_end/1000.0:.2f}s)")
            
            chunk.export(
                chunk_path,
                format="mp3",
                parameters=[
                    "-ac", "1",
                    "-ar", "16000",
                    "-b:a", "64k",
                    "-write_xing", "0",
                    "-q:a", "9"
                ]
            )
            
            chunks.append({
                'path': chunk_path,
                'start_time': current_position / 1000.0,
                'end_time': end_position / 1000.0,
                'duration': (end_position - current_position) / 1000.0,
                'overlap_start': chunk_start / 1000.0,
                'overlap_end': chunk_end / 1000.0,
                'has_previous': current_position > 0,
                'has_next': end_position < duration_ms
            })
            
            current_position = end_position
            chunk_number += 1
        
        return chunks, temp_dir
        
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


def transcribe_audio_chatgpt(client: OpenAI, audio_path: str, video_duration: float) -> str:
    """Transcription function with improved timing accuracy and chunk handling"""
    audio_mp3_path = extract_audio(audio_path)
    temp_dir = None
    
    try:
        all_segments = []
        
        # Process large files in chunks
        if os.path.getsize(audio_mp3_path) > 1024 * 1024 * 24:
            print("Audio file larger than 25MB, splitting into chunks...")
            chunks, temp_dir = split_audio(audio_mp3_path, video_duration)
            
            for chunk in chunks:
                print(f"Processing chunk {chunks.index(chunk) + 1}/{len(chunks)}...")
                
                with open(chunk['path'], "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                    
                    # Process segments with precise timing
                    processed_segments = process_chunk_segments(chunk, response.segments, video_duration)
                    all_segments.extend(processed_segments)
                    
                # Brief pause between chunks
                time.sleep(0.5)
        else:
            # Process single file
            with open(audio_mp3_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                all_segments = response.segments
                
                # Validate timestamps
                for segment in all_segments:
                    segment.start = min(float(segment.start), video_duration)
                    segment.end = min(float(segment.end), video_duration)
        
        # Sort and validate segments
        all_segments.sort(key=lambda x: x.start)
        validate_segments_continuity(all_segments)
        
        # Generate final transcript
        transcript = ""
        for i, segment in enumerate(all_segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            transcript += f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
        
        print(f"Total segments: {len(all_segments)}")
        if all_segments:
            print(f"First segment starts at: {format_timestamp(all_segments[0].start)}")
            print(f"Last segment ends at: {format_timestamp(all_segments[-1].end)}")
            print(f"Total duration: {all_segments[-1].end - all_segments[0].start:.2f} seconds")
        
        return transcript
        
    finally:
        if os.path.exists(audio_mp3_path):
            os.remove(audio_mp3_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def validate_segments_continuity(segments: List) -> None:
    """Ensure continuous timing between segments with improved validation"""
    if not segments:
        return
        
    from decimal import Decimal, ROUND_HALF_UP
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        # Convert to Decimal for precise comparison
        curr_end = Decimal(str(current.end))
        next_start = Decimal(str(next_seg.start))
        
        # Fix gaps (allow 300ms gap)
        if next_start - curr_end > Decimal('0.3'):
            print(f"Found gap between segments {i} and {i+1}, adjusting...")
            middle = (curr_end + next_start) / 2
            middle = middle.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
            current.end = float(middle)
            next_seg.start = float(middle)
        
        # Fix overlaps
        elif curr_end > next_start:
            print(f"Found overlap between segments {i} and {i+1}, adjusting...")
            middle = (curr_end + next_start) / 2
            middle = middle.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
            current.end = float(middle)
            next_seg.start = float(middle)

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
    
    # Check if ffmpeg is installed
    if not shutil.which('ffmpeg'):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'libmp3lame',
            '-ac', '1',
            '-ar', '16000',
            '-b:a', '64k',        # Control bitrate
            '-filter:a', 'dynaudnorm=f=150:g=15',  # Normalize audio
            '-write_xing', '0',
            '-y',
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
    
def format_timestamp(seconds: float, precision: int = 3) -> str:
    """Format timestamp with controlled precision and decimal arithmetic"""
    # Convert to Decimal for precise arithmetic
    seconds_dec = Decimal(str(seconds)).quantize(
        Decimal('0.001'), 
        rounding=ROUND_HALF_UP
    )
    
    hours = int(seconds_dec // Decimal('3600'))
    minutes = int((seconds_dec % Decimal('3600')) // Decimal('60'))
    remaining_seconds = seconds_dec % Decimal('60')
    
    # Format with exact precision
    decimal_format = f"{{:0{precision + 3}.{precision}f}}"
    formatted_seconds = decimal_format.format(float(remaining_seconds))
    
    return f"{hours:02d}:{minutes:02d}:{formatted_seconds}".replace(".", ",")

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

def merge_short_segments(segments: List, min_words: int = 5) -> List:
    """Merge segments that are too short with adjacent segments.
    
    Args:
        segments: List of segment dictionaries
        min_words: Minimum number of words a segment should have
    """
    merged = []
    i = 0
    
    while i < len(segments):
        current = segments[i]
        word_count = len(current['text'].split())
        
        # If this is a short segment
        if word_count < min_words:
            # Try to merge with the previous segment first
            if merged and i > 0:
                prev = merged[-1]
                # Merge with previous segment
                merged[-1] = {
                    'number': prev['number'],
                    'start': prev['start'],
                    'end': current['end'],
                    'text': prev['text'] + ' ' + current['text']
                }
            # If we can't merge with previous, try to merge with next
            elif i + 1 < len(segments):
                next_seg = segments[i + 1]
                merged.append({
                    'number': current['number'],
                    'start': current['start'],
                    'end': next_seg['end'],
                    'text': current['text'] + ' ' + next_seg['text']
                })
                i += 1  # Skip the next segment since we merged it
            else:
                # If we can't merge with anything, keep it as is
                merged.append(current)
        else:
            merged.append(current)
        
        i += 1
    
    # Renumber segments sequentially
    for idx, segment in enumerate(merged, start=1):
        segment['number'] = idx
    
    return merged


def translate_bulk(client, segments, target_language):
    """Translate segments with enhanced logging, debugging, and partial success handling"""
    
    # Format segments with clear boundaries and validation markers
    text_to_translate = ""
    for seg in segments:
        # Add boundary markers to help with parsing
        text_to_translate += f"[START_SEG{seg['number']}]\n{seg['text']}\n[END_SEG{seg['number']}]\n\n"
    
    # Log the exact input being sent to the API
    print(f"\nDEBUG: Translation Input:")
    print(f"Number of segments to translate: {len(segments)}")
    print(f"First segment number: {segments[0]['number']}")
    print(f"Last segment number: {segments[-1]['number']}")
    print(f"Sample of text being sent:\n{text_to_translate[:500]}...")
    
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
- Keep all [START_SEG#] and [END_SEG#] markers exactly as they appear
- Maintain exact segment numbering
- Place your translation between the START and END markers
- Do not add any additional text or explanations
- Keep one empty line between segments

Example format:
[START_SEG1]
¿Qué tal?
[END_SEG1]

[START_SEG2]
¿Cómo estás?
[END_SEG2]"""
        },
        {
            "role": "user", 
            "content": text_to_translate
        }
    ]
    
    successful_translations = {}  # Track successful translations
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"\nDEBUG: Attempt {attempt + 1}/{max_retries}")
            print(f"Input token count: {len(text_to_translate.split())}")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Log API response characteristics
            print(f"\nDEBUG: Translation Response:")
            print(f"Response length: {len(translated_text)}")
            print(f"Sample of response:\n{translated_text[:500]}...")
            
            # Parse segments with robust error handling
            current_translations = parse_translated_segments(translated_text, len(segments))
            
            # Log parsing results
            print(f"\nDEBUG: Parsed Segments:")
            print(f"Number of parsed segments: {len(current_translations)}")
            print(f"Segment numbers found: {sorted(current_translations.keys())}")
            
            # Add successful translations to our collection
            for seg_num, translation in current_translations.items():
                if translation and translation.strip():
                    successful_translations[int(seg_num)] = translation
            
            # Check for missing segments
            original_numbers = {seg['number'] for seg in segments}
            translated_numbers = set(successful_translations.keys())
            missing = original_numbers - translated_numbers
            
            if missing:
                print(f"\nDEBUG: Missing segments analysis:")
                print(f"Missing segment numbers: {missing}")
                
                # If this isn't the last attempt, prepare for retry with just missing segments
                if attempt < max_retries - 1:
                    # Create new text_to_translate with only missing segments
                    retry_segments = [seg for seg in segments if seg['number'] in missing]
                    text_to_translate = ""
                    for seg in retry_segments:
                        text_to_translate += f"[START_SEG{seg['number']}]\n{seg['text']}\n[END_SEG{seg['number']}]\n\n"
                    
                    print(f"\nRetrying {len(missing)} segments in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            
            # If we have all segments or this is our last attempt, return what we have
            if not missing or attempt == max_retries - 1:
                return successful_translations
                
        except Exception as e:
            print(f"\nDEBUG: Exception during translation:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            
            if attempt < max_retries - 1:
                # For exceptions, retry with all untranslated segments
                missing = {seg['number'] for seg in segments} - set(successful_translations.keys())
                if missing:
                    retry_segments = [seg for seg in segments if seg['number'] in missing]
                    text_to_translate = ""
                    for seg in retry_segments:
                        text_to_translate += f"[START_SEG{seg['number']}]\n{seg['text']}\n[END_SEG{seg['number']}]\n\n"
                
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # On final attempt, return what we have
                return successful_translations
    
    raise RuntimeError("Failed to obtain valid translation after all retries")



def validate_translation_response(translated_segments, original_segments):
    """Validate the translation response for completeness and integrity"""
    
    # Check for missing segments
    original_numbers = {seg['number'] for seg in original_segments}
    translated_numbers = set(translated_segments.keys())
    missing_segments = original_numbers - translated_numbers
    
    if missing_segments:
        print(f"Warning: Missing segments: {missing_segments}")
        return False
    
    # Check for empty or invalid translations
    for seg_num, content in translated_segments.items():
        if not content or len(content.strip()) == 0:
            print(f"Warning: Empty translation for segment {seg_num}")
            return False
    
    # Validate segment order
    if list(translated_segments.keys()) != sorted(translated_segments.keys()):
        print("Warning: Segments are not in sequential order")
        return False
    
    return True

def parse_translated_segments(translated_text, expected_count):
    """Parse translated segments with enhanced logging"""
    segments = {}
    
    print(f"\nDEBUG: Parsing Translation Response")
    print(f"Expected segment count: {expected_count}")
    
    # Split into chunks by START_SEG markers
    chunks = re.split(r'\[START_SEG(\d+)\]', translated_text)
    chunks = chunks[1:]  # Remove initial empty chunk
    
    print(f"Number of chunks found: {len(chunks)}")
    
    if len(chunks) == 0:
        print("ERROR: No valid segments found in translation")
        return segments
    
    # Process chunks in pairs
    for i in range(0, len(chunks), 2):
        if i + 1 >= len(chunks):
            print(f"WARNING: Odd number of chunks, skipping last chunk")
            break
            
        try:
            segment_num = int(chunks[i])
            content = chunks[i + 1]
            
            print(f"\nProcessing segment {segment_num}:")
            print(f"Content length: {len(content)}")
            
            # Extract content between START and END markers
            match = re.search(r'(.*?)\[END_SEG\d+\]', content, re.DOTALL)
            if match:
                segment_text = match.group(1).strip()
                if segment_text:
                    segments[segment_num] = segment_text
                    print(f"Successfully parsed segment {segment_num}")
                else:
                    print(f"WARNING: Empty content for segment {segment_num}")
            else:
                print(f"WARNING: Could not find END_SEG marker for segment {segment_num}")
                print(f"Content sample: {content[:100]}...")
            
        except (ValueError, AttributeError) as e:
            print(f"ERROR parsing segment {i//2 + 1}: {str(e)}")
            print(f"Chunk content: {chunks[i][:100]}...")
            continue
    
    print(f"\nTotal segments parsed: {len(segments)}")
    return segments

def process_transcript(transcript, target_language, api_key):
    client = OpenAI(api_key=api_key)
    if not client.api_key:
        raise ValueError("Invalid OpenAI API key.")

    # Extract and merge segments as before
    segments = extract_segments(transcript)
    if not segments:
        raise ValueError("No segments found in transcript")
    
    print("Merging short segments...")
    original_count = len(segments)
    segments = merge_short_segments(segments)
    print(f"Merged {original_count - len(segments)} segments")
    
    # Translation state tracking
    successful_translations = {}  # Store successful translations by segment number
    remaining_segments = segments.copy()  # Track segments still needing translation
    retry_count = {}  # Track retry attempts per segment
    
    def calculate_batch_size(segments_to_process, max_tokens=3000):
        """Calculate optimal batch size based on text length"""
        avg_tokens_per_segment = max(
            len(s['text'].split()) * 1.5  # Conservative token estimate
            for s in segments_to_process
        )
        return max(1, min(10, int(max_tokens / avg_tokens_per_segment)))
    
    def process_batch(batch, attempt=1):
        """Process a batch with detailed logging"""
        print(f"\nProcessing batch of {len(batch)} segments (Attempt {attempt})")
        try:
            translations = translate_bulk(client, batch, target_language)
            return translations
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            return None
    
    def handle_failed_segments(failed_segments, max_retries=3):
        """Handle failed segments with progressively smaller batches"""
        if not failed_segments:
            return {}
            
        print(f"\nRetrying {len(failed_segments)} failed segments...")
        retry_translations = {}
        
        # Sort failed segments by retry count
        segments_by_retries = {}
        for seg in failed_segments:
            retry_count.setdefault(seg['number'], 0)
            retry_count[seg['number']] += 1
            if retry_count[seg['number']] <= max_retries:
                segments_by_retries.setdefault(retry_count[seg['number']], []).append(seg)
        
        # Process each retry group
        for retry_num, retry_segments in segments_by_retries.items():
            # Use smaller batch size for retries
            batch_size = max(1, calculate_batch_size(retry_segments) // 2)
            print(f"\nRetry #{retry_num} with batch size {batch_size}")
            
            # Process in smaller batches
            for i in range(0, len(retry_segments), batch_size):
                batch = retry_segments[i:i + batch_size]
                translations = process_batch(batch, attempt=retry_num)
                
                if translations:
                    for seg_num, translation in translations.items():
                        retry_translations[seg_num] = translation
        
        return retry_translations
    
    # Initial processing
    batch_size = calculate_batch_size(segments)
    print(f"\nInitial processing with batch size: {batch_size}")
    
    while remaining_segments:
        current_batch = remaining_segments[:batch_size]
        remaining_segments = remaining_segments[batch_size:]
        
        # Process current batch
        translations = process_batch(current_batch)
        if translations:
            successful_translations.update(translations)
            
            # Identify failed segments in this batch
            failed_segments = [
                seg for seg in current_batch
                if seg['number'] not in translations
            ]
            
            # Handle failures if any
            if failed_segments:
                retry_translations = handle_failed_segments(failed_segments)
                successful_translations.update(retry_translations)
                
                # Add any segments that still failed to remaining_segments
                still_failed = [
                    seg for seg in failed_segments
                    if seg['number'] not in retry_translations
                ]
                remaining_segments.extend(still_failed)
    
    # Generate final SRT content
    translated_srt = ""
    translated_segments_with_timing = []
    
    for segment in segments:
        translation = successful_translations.get(segment['number'])
        if translation and translation != "[Translation error]":
            translated_srt += f"{segment['number']}\n"
            translated_srt += f"{segment['start']} --> {segment['end']}\n"
            translated_srt += f"{translation}\n\n"
            
            translated_segments_with_timing.append({
                'number': segment['number'],
                'start': segment['start'],
                'end': segment['end'],
                'text': translation
            })
    
    # Report final status
    print(f"\nTranslation Summary:")
    print(f"Total segments: {len(segments)}")
    print(f"Successfully translated: {len(translated_segments_with_timing)}")
    print(f"Failed segments: {len(segments) - len(translated_segments_with_timing)}")
    
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

def get_video_duration_seconds(video_path):
    """Get duration of video in seconds using ffprobe"""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=None if os.name == 'nt' else None
        )
        
        stdout_data, _ = process.communicate()
        duration_info = json.loads(stdout_data)
        return float(duration_info['format']['duration'])
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        raise

def check_ffmpeg_availability():
    """Check if ffmpeg is available in the system path"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def calculate_total_frames(video_path):
    """Calculate total frames in video using ffprobe"""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames',
            '-of', 'json',
            video_path
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout_data, _ = process.communicate()
        stream_info = json.loads(stdout_data)
        
        # Try to get nb_frames first
        if 'streams' in stream_info and stream_info['streams']:
            nb_frames = stream_info['streams'][0].get('nb_frames')
            if nb_frames and nb_frames != 'N/A':
                return int(nb_frames)
        
            # If nb_frames not available, calculate from duration and frame rate
            r_frame_rate = stream_info['streams'][0].get('r_frame_rate', '')
            if r_frame_rate:
                # Parse frame rate fraction (e.g., "30/1")
                num, den = map(int, r_frame_rate.split('/'))
                fps = num / den
                
                # Get duration and calculate total frames
                duration = get_video_duration(video_path)
                return int(duration * fps)
                
    except Exception as e:
        print(f"Warning: Could not calculate exact frame count: {str(e)}")
        # Provide an estimate based on duration assuming 30fps
        duration = get_video_duration(video_path)
        return int(duration * 30)
    
def add_subtitles_with_ffmpeg(video_path, srt_path, output_path, font='NanumGothic'):
    """Add subtitles to video using ffmpeg with progress bar"""
    print(f"Adding subtitles to video...")
    
    try:
        # Get video dimensions and duration
        width, height = get_video_dimensions(video_path)
        duration = get_video_duration_seconds(video_path)
        font_size = calculate_font_size(width)
        
        print(f"Video dimensions: {width}x{height}")
        print(f"Video duration: {duration:.2f} seconds")
        
        # Calculate total frames for progress tracking
        total_frames = calculate_total_frames(video_path)
        print(f"Total frames to process: {total_frames}")
        
        # Prepare file paths
        def prepare_path(path):
            path = path.replace('\\', '/')
            if ' ' in path or '(' in path or ')' in path:
                path = f"'{path}'"
            return path
            
        video_path = prepare_path(video_path)
        srt_path = prepare_path(srt_path)
        output_path = prepare_path(output_path)
        
        # Construct subtitle filter
        subtitle_filter = f"subtitles={srt_path}"
        style = f":force_style='Fontname={font},FontSize={font_size},PrimaryColour=&HFFFFFF,OutlineColour=&H40000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=35'"
        
        # Create command
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f"{subtitle_filter}{style}",
            '-c:a', 'copy',
            '-y',
            output_path
        ]
        
        # Start process with output capture
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Setup progress bar
        with tqdm(total=total_frames, unit='frames', desc="Processing", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}]') as pbar:
            current_frame = 0
            
            # Monitor output and update progress
            while True:
                line = process.stderr.readline()
                
                if line == '' and process.poll() is not None:
                    break
                    
                if line:
                    # Parse frame count from ffmpeg output
                    frame_match = re.search(r'frame=\s*(\d+)', line)
                    if frame_match:
                        new_frame = int(frame_match.group(1))
                        if new_frame > current_frame:
                            increment = new_frame - current_frame
                            pbar.update(increment)
                            current_frame = new_frame
            
            # Get final return code
            rc = process.poll()
            if rc != 0:
                stderr = process.stderr.read()
                raise RuntimeError(f"ffmpeg process failed with error:\n{stderr}")
        
        print(f"\nVideo with translated subtitles saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError adding subtitles: {str(e)}")
        raise
    
def get_video_duration(video_path):
    """Get exact video duration using ffprobe"""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout_data, _ = process.communicate()
        duration_info = json.loads(stdout_data)
        return float(duration_info['format']['duration'])
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
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
    video_duration = get_video_duration(video_path)
    print(f"Video duration: {video_duration:.2f} seconds")
    
    # Modified transcription call
    if args.use_local_whisper:
        print("Using local whisper model")
        transcript = transcribe_audio(video_path, args.models_path)
    else:
        print("Using OpenAI API whisper model")
        transcript = transcribe_audio_chatgpt(client, video_path, video_duration)
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