# YouTube Video Processor

Downloads YouTube videos and extracts audio, first frame, and isolated voice tracks using AI-powered source separation.

## Features
- Download YouTube videos in highest quality
- Extract audio as WAV files
- Extract first frame as JPEG
- Isolate voice tracks using Demucs ML model (with frequency filtering fallback)

## Usage

```bash
# Install dependencies
uv sync

# Process a YouTube video
uv run youtube_processor.py "https://youtube.com/watch?v=VIDEO_ID"

# Specify custom output directory
uv run youtube_processor.py "https://youtube.com/watch?v=VIDEO_ID" --output-dir custom_output

# Run tests
uv run pytest
```

## Output Files
- `{title}.mp4` - Downloaded video
- `{title}_audio.wav` - Extracted audio
- `{title}_first_frame.jpg` - First video frame
- `{title}_audio_voice_isolated.wav` - Isolated voice track