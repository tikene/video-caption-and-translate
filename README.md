# Video Transcriber and Translator
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script that automates video subtitle creation & translation, supporting both local files and online video URLs.

## 🚀 Features
- Download videos from various platforms easily by specifying the URL (YouTube, Twitter, Facebook, Instagram, etc.)
- Transcribe audio using ChatGPT's Whisper model (default) or local Whisper model
- Translate the generated transcriptions to a target language using OpenAI's GPT models
- Add translated subtitles to videos
- Support for multiple languages and resolutions (including YouTube Shorts)
- Support for long videos, these are split into chunks automatically before transcribing and translating

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
- `--use_local_whisper`: Use local Whisper model for transcription instead of ChatGPT's Whisper

## 📋 Examples
1. Translate YouTube video subtitles to Spanish (using default ChatGPT Whisper):
   ```bash
   python translate.py https://www.youtube.com/watch?v=VIDEO_ID Spanish
   ```

2. Translate local video file subtitles to French (using default ChatGPT Whisper):
   ```bash
   python translate.py /path/to/your/video.mp4 French
   ```

3. Use a specific output directory and font:
   ```bash
   python translate.py input_video.mp4 German --output_dir my_output --font Arial
   ```

4. Use a local model for transcription:
   ```bash
   python translate.py input_video.mp4 Korean --use_local_whisper
   ```

## 🛠️ Installation

### 1. FFmpeg Installation

#### Windows:
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the ZIP file
3. Add the `bin` folder path to system PATH
4. Verify installation: `ffmpeg -version`

#### macOS:
```bash
brew install ffmpeg
```

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

### 2. Python Dependencies
Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

Install required packages:

```bash
pip install openai==1.12.0
pip install faster-whisper==0.10.0
pip install yt-dlp==2024.3.10
pip install ffmpeg-python==0.2.0
pip install pydub==0.25.1
```

### 3. OpenAI API Setup
1. Create account at [platform.openai.com](https://platform.openai.com)
2. Generate API key in account settings
3. Set environment variable:
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY='your-key-here'
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY='your-key-here'
   ```

### 4. Download the repository

1. Clone repository:
```bash
git clone https://github.com/tikene/video-caption-and-translate.git
cd video-caption-and-translate
```

2. Verify installation:
```bash
python translate.py --help
```

## ⏳ Common Issues

### FFmpeg Not Found
- Ensure FFmpeg is in system PATH
- Restart terminal/IDE after PATH changes
- Check with `ffmpeg -version`

### OpenAI API Errors
- Verify API key is set correctly
- Check account has sufficient credits
- Ensure stable internet connection

## 📂 Output
The script generates the following files in the output directory:
1. Downloaded video (if URL was provided)
2. Translated SRT subtitle file
3. Video with embedded translated subtitles

## ⚠️ Important Notes
- The script uses the GPT-4 model for translation by default, which costs around $0.1 cents for a two-minute video. You may reduce token costs by switching to 'gpt-4o-mini' which is ~80% cheaper at the expense of translation quality
- Longer video -> Higher costs (duh)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
