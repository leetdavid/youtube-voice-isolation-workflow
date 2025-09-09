#!/usr/bin/env python3
"""
YouTube Video Processor
Downloads YouTube videos, extracts audio as WAV, first frame as JPEG, and isolates voice.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Tuple
import cv2
import torch
import torchaudio
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter
import yt_dlp
from demucs.pretrained import get_model
from demucs.apply import apply_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeProcessor:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._demucs_model = None
        
    def _get_demucs_model(self):
        """Load Demucs model for voice separation (lazy loading)."""
        if self._demucs_model is None:
            logger.info("Loading Demucs model for voice separation...")
            self._demucs_model = get_model('htdemucs')
            self._demucs_model.eval()
        return self._demucs_model
        
    def download_video(self, url: str) -> Tuple[str, str]:
        """Download YouTube video in highest quality and return video and audio paths."""
        with tempfile.TemporaryDirectory():
            video_path = os.path.join(self.output_dir, "%(title)s.%(ext)s")
            audio_path = os.path.join(self.output_dir, "%(title)s_audio.%(ext)s")
            
            # Download video
            video_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': video_path,
                'merge_output_format': 'mp4',
            }
            
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_filename = ydl.prepare_filename(info)
                if not video_filename.endswith('.mp4'):
                    video_filename = video_filename.rsplit('.', 1)[0] + '.mp4'
                    
            # Download audio separately for better quality
            audio_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': audio_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
            }
            
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([url])
                base_filename = ydl.prepare_filename(info)
                audio_filename = base_filename.rsplit('.', 1)[0] + '_audio.wav'
                
        logger.info(f"Downloaded video: {video_filename}")
        logger.info(f"Extracted audio: {audio_filename}")
        
        return video_filename, audio_filename
    
    def extract_first_frame(self, video_path: str) -> str:
        """Extract the first frame of the video as JPEG."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not read first frame from video")
        
        # Generate output filename
        video_name = Path(video_path).stem
        frame_path = self.output_dir / f"{video_name}_first_frame.jpg"
        
        # Save frame as JPEG
        cv2.imwrite(str(frame_path), frame)
        logger.info(f"Extracted first frame: {frame_path}")
        
        return str(frame_path)
    
    def isolate_voice(self, audio_path: str) -> str:
        """Isolate voice from audio using state-of-the-art ML model (Demucs)."""
        logger.info("Starting voice isolation using Demucs ML model...")
        
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"Loaded audio: {waveform.shape}, sample_rate: {sample_rate}")
        
        # Ensure we have the correct sample rate for Demucs (44100 Hz)
        if sample_rate != 44100:
            logger.info(f"Resampling from {sample_rate} to 44100 Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
            sample_rate = 44100
        
        # Convert to stereo if mono (Demucs expects stereo)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            # Take first 2 channels if more than stereo
            waveform = waveform[:2]
        
        # Get the Demucs model
        model = self._get_demucs_model()
        
        # Apply source separation
        logger.info("Running source separation...")
        with torch.no_grad():
            # Add batch dimension
            waveform_batch = waveform.unsqueeze(0)
            
            # Apply the model
            sources = apply_model(model, waveform_batch, device='cpu', progress=True)[0]
            
            # Demucs htdemucs outputs: [drums, bass, other, vocals]
            vocals = sources[3]  # Extract vocals
            
        logger.info("Voice separation completed!")
        
        # Generate output filename
        audio_name = Path(audio_path).stem
        voice_path = self.output_dir / f"{audio_name}_voice_isolated.wav"
        
        # Save the isolated vocals
        torchaudio.save(str(voice_path), vocals, sample_rate)
        logger.info(f"Isolated voice saved: {voice_path}")
        
        return str(voice_path)
    
    def isolate_voice_fallback(self, audio_path: str) -> str:
        """Fallback voice isolation using frequency filtering (legacy method)."""
        logger.info("Using fallback frequency filtering method...")
        
        audio = AudioSegment.from_wav(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Apply frequency filtering to isolate human voice range (80Hz - 8kHz)
        audio_filtered = high_pass_filter(audio, cutoff=80)
        audio_filtered = low_pass_filter(audio_filtered, cutoff=8000)
        
        # Normalize audio
        audio_filtered = audio_filtered.normalize()
        
        # Apply compression to even out volume levels
        audio_filtered = audio_filtered.compress_dynamic_range()
        
        # Generate output filename
        audio_name = Path(audio_path).stem
        voice_path = self.output_dir / f"{audio_name}_voice_isolated_fallback.wav"
        
        # Export processed audio
        audio_filtered.export(str(voice_path), format="wav")
        logger.info(f"Isolated voice (fallback): {voice_path}")
        
        return str(voice_path)
    
    def process_youtube_url(self, url: str) -> dict:
        """Process a YouTube URL and return paths to all generated files."""
        logger.info(f"Processing YouTube URL: {url}")
        
        # Download video and extract audio
        video_path, audio_path = self.download_video(url)
        
        # Extract first frame
        frame_path = self.extract_first_frame(video_path)
        
        # Isolate voice from audio using ML model with fallback
        try:
            voice_path = self.isolate_voice(audio_path)
        except Exception as e:
            logger.warning(f"ML voice isolation failed: {e}")
            logger.info("Falling back to frequency filtering method...")
            voice_path = self.isolate_voice_fallback(audio_path)
        
        result = {
            'video_path': video_path,
            'audio_path': audio_path,
            'frame_path': frame_path,
            'voice_path': voice_path
        }
        
        logger.info("Processing completed successfully!")
        logger.info(f"Results: {result}")
        
        return result


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process YouTube videos")
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    processor = YouTubeProcessor(args.output_dir)
    result = processor.process_youtube_url(args.url)
    
    print("\n=== Processing Results ===")
    for key, path in result.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
