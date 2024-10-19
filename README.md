# Video Transcriber and Translator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script that automates video subtitle creation & translation, supporting both local files and online video URLs.

<br>

## 🚀 Features

- Download videos from various platforms (YouTube, Twitter, Facebook, Instagram, etc.)
- Transcribe audio using Whisper or ChatGPT's Whisper model
- Translate transcriptions to a target language using OpenAI's GPT models
- Add translated subtitles to videos
- Support for multiple languages and resolutions (including YouTube Shorts)


## 🎬 Demo

Here are some examples of the script in action:

1. **English to Korean Translation**

   ![American Psycho - Korean](https://github.com/user-attachments/assets/5c76cd45-6221-4ef1-a6bc-367affa5dbe6)

2. **English to Spanish Translation**

   ![Meet the Spy - Spanish](https://github.com/user-attachments/assets/284a9e8d-1fd6-4fbf-bcdb-24a8e284d32f)

3. **German to English Translation**

   ![Adolf Hitler Speech - German](https://github.com/user-attachments/assets/76d67ac5-d5a7-46b2-8a24-addb8dff24af)

4. **Spanish to English Translation**

   ![El hoyo - Spanish](https://github.com/user-attachments/assets/50a33248-83c6-49a7-8d50-42eb735dfe87)

**Note**: Results may vary with accents or background music, especially when using the local model. Video and caption synchronization might be affected when the audio isn't clear. The above clips were processed using OpenAI for both transcription & translation (default script behavior).


## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tikene/video-caption-and-translate.git
   cd video-caption-and-translate
   ```

2. Install required dependencies:
   ```bash
   pip install openai faster_whisper yt-dlp ffmpeg-python
   ```

3. Install FFmpeg:
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add to your system PATH

4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```


## 🖥️ Usage

Run the script using the following command:

```bash
python translate.py video_input target_language [options]
```

### Arguments:
- `video_input`: URL or path to the input video file
- `target_language`: Target language for translation (e.g., Spanish, English, French)

### Options:
- `--output_dir`: Directory to save output files (default: "output")
- `--models_path`: Path to store Whisper models (default: "Models")
- `--openai_api_key`: OpenAI API key (if not set as an environment variable)
- `--font`: Font to use for subtitles (default: "NanumGothic")
- `--use_chatgpt_whisper`: Use ChatGPT's Whisper model for transcription (default: True)


## 📋 Examples

1. Translate YouTube video subtitles to Spanish:
   ```bash
   python translate.py https://www.youtube.com/watch?v=VIDEO_ID Spanish
   ```

2. Translate local video file subtitles to French:
   ```bash
   python translate.py /path/to/your/video.mp4 French
   ```

3. Use a specific output directory and font:
   ```bash
   python translate.py input_video.mp4 German --output_dir my_output --font Arial
   ```

4. Use a local model for transcription:
   ```bash
   python translate.py input_video.mp4 Korean --use_chatgpt_whisper False
   ```


## 📂 Output

The script generates the following files in the output directory:
1. Downloaded video (if URL was provided)
2. Translated SRT subtitle file
3. Video with embedded translated subtitles


## ⚠️ Important Notes

- Ensure you have a valid OpenAI API key with sufficient credits.
- The script uses the GPT-4o model for translation, which may incur costs on your OpenAI account (approximately 1-3 cents for a two-minute video).
- Large videos may not work and/or significantly increase costs. Testing has been limited to short videos (up to 5 minutes in length).



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
