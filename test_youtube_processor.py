"""
Tests for YouTube video processor.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import cv2
import numpy as np
from pydub import AudioSegment

from youtube_processor import YouTubeProcessor


class TestYouTubeProcessor:
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance with temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield YouTubeProcessor(temp_dir)
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a sample video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a simple test video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 1.0, (100, 100))
            
            # Write a few frames
            for i in range(5):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                frame[:] = [i*50, i*50, i*50]  # Gray gradient
                out.write(frame)
            
            out.release()
            yield f.name
            
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create a simple sine wave using AudioSegment's generator
            duration = 1000  # 1 second in ms
            sample_rate = 44100
            frequency = 440  # A4 note
            
            # Generate sine wave using AudioSegment
            audio = AudioSegment.silent(duration, frame_rate=sample_rate)
            
            # Create a sine wave tone
            import math
            samples = []
            for i in range(int(sample_rate * duration / 1000)):
                t = i / sample_rate
                sample = int(16383 * math.sin(2 * math.pi * frequency * t))
                samples.append(sample)
            
            # Convert to bytes
            sample_bytes = b''.join([sample.to_bytes(2, byteorder='little', signed=True) for sample in samples])
            
            # Create AudioSegment from raw audio data
            audio = AudioSegment(
                sample_bytes,
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            audio.export(f.name, format="wav")
            yield f.name
            
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass
    
    def test_init(self):
        """Test processor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = YouTubeProcessor(temp_dir)
            assert processor.output_dir == Path(temp_dir)
            assert processor.output_dir.exists()
    
    def test_init_creates_output_dir(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "new_output"
            YouTubeProcessor(str(output_dir))
            assert output_dir.exists()
    
    @patch('youtube_processor.yt_dlp.YoutubeDL')
    def test_download_video(self, mock_yt_dlp, processor):
        """Test video download functionality."""
        # Mock the YoutubeDL context manager and its methods
        mock_ydl_instance = Mock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        
        # Mock extract_info to return sample video info
        mock_info = {
            'title': 'Test Video',
            'ext': 'mp4'
        }
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl_instance.prepare_filename.return_value = str(processor.output_dir / "Test Video.mp4")
        
        # Test download
        url = "https://www.youtube.com/watch?v=test"
        video_path, audio_path = processor.download_video(url)
        
        # Verify calls
        assert mock_yt_dlp.call_count == 2  # Once for video, once for audio
        mock_ydl_instance.extract_info.assert_called()
        mock_ydl_instance.download.assert_called_once_with([url])
        
        # Verify paths
        assert "Test Video" in video_path
        assert video_path.endswith('.mp4')
        assert "Test Video" in audio_path
        assert audio_path.endswith('.wav')
    
    def test_extract_first_frame(self, processor, sample_video_file):
        """Test first frame extraction."""
        frame_path = processor.extract_first_frame(sample_video_file)
        
        # Verify file was created
        assert os.path.exists(frame_path)
        assert frame_path.endswith('.jpg')
        
        # Verify it's a valid image
        img = cv2.imread(frame_path)
        assert img is not None
        assert img.shape[2] == 3  # RGB channels
    
    def test_extract_first_frame_invalid_video(self, processor):
        """Test first frame extraction with invalid video file."""
        with pytest.raises(ValueError, match="Could not open video file"):
            processor.extract_first_frame("nonexistent_file.mp4")
    
    @patch('youtube_processor.apply_model')
    @patch('youtube_processor.get_model')
    def test_isolate_voice_ml(self, mock_get_model, mock_apply_model, processor, sample_audio_file):
        """Test ML-based voice isolation functionality."""
        import torch
        
        # Mock the ML model
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        # Mock the model output (drums, bass, other, vocals)
        # Create mock output with 4 source tracks
        sample_length = 44100  # 1 second of audio
        mock_sources = torch.randn(4, 2, sample_length)  # 4 sources, stereo, 1 second
        mock_apply_model.return_value = [mock_sources]
        
        voice_path = processor.isolate_voice(sample_audio_file)
        
        # Verify file was created
        assert os.path.exists(voice_path)
        assert voice_path.endswith('.wav')
        
        # Verify model was called
        mock_get_model.assert_called_once_with('htdemucs')
        mock_apply_model.assert_called_once()
        
        # Verify it's a valid audio file
        import torchaudio
        waveform, sample_rate = torchaudio.load(voice_path)
        assert waveform.shape[0] == 2  # Should be stereo (from Demucs output)
        assert sample_rate == 44100
    
    def test_isolate_voice_fallback(self, processor, sample_audio_file):
        """Test fallback voice isolation functionality."""
        voice_path = processor.isolate_voice_fallback(sample_audio_file)
        
        # Verify file was created
        assert os.path.exists(voice_path)
        assert voice_path.endswith('.wav')
        
        # Verify it's a valid audio file
        audio = AudioSegment.from_wav(voice_path)
        assert len(audio) > 0
        assert audio.channels == 1  # Should be mono
        assert audio.frame_rate > 0
    
    def test_isolate_voice_fallback_stereo_to_mono(self, processor):
        """Test voice isolation converts stereo to mono."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create stereo audio
            duration = 1000
            sample_rate = 44100
            frequency = 440
            
            # Generate stereo sine wave
            import math
            stereo_samples = []
            for i in range(int(sample_rate * duration / 1000)):
                t = i / sample_rate
                sample = int(16383 * math.sin(2 * math.pi * frequency * t))
                stereo_samples.append(sample)  # Left channel
                stereo_samples.append(sample)  # Right channel
            
            # Convert to bytes
            sample_bytes = b''.join([sample.to_bytes(2, byteorder='little', signed=True) for sample in stereo_samples])
            
            stereo_audio = AudioSegment(
                sample_bytes,
                frame_rate=sample_rate,
                sample_width=2,
                channels=2
            )
            stereo_audio.export(f.name, format="wav")
            
            # Test conversion using fallback method
            voice_path = processor.isolate_voice_fallback(f.name)
            processed_audio = AudioSegment.from_wav(voice_path)
            
            assert processed_audio.channels == 1
            
            # Cleanup
            os.unlink(f.name)
    
    @patch('youtube_processor.get_model')
    def test_isolate_voice_ml_failure_fallback(self, mock_get_model, processor, sample_audio_file):
        """Test that ML failure triggers fallback to frequency filtering."""
        # Make the ML model fail
        mock_get_model.side_effect = Exception("Model loading failed")
        
        # Call through the main process method which has error handling
        with patch.object(processor, 'download_video', return_value=('video.mp4', sample_audio_file)), \
             patch.object(processor, 'extract_first_frame', return_value='frame.jpg'):
            result = processor.process_youtube_url("https://test.com")
        
        # Verify fallback file was created
        voice_path = result['voice_path']
        assert os.path.exists(voice_path)
        assert voice_path.endswith('.wav')
        
        # Since it used fallback, it should be mono
        audio = AudioSegment.from_wav(voice_path)
        assert audio.channels == 1
    
    @patch('youtube_processor.yt_dlp.YoutubeDL')
    def test_process_youtube_url_integration(self, mock_yt_dlp, processor, sample_video_file, sample_audio_file):
        """Test the complete processing workflow."""
        # Mock YoutubeDL
        mock_ydl_instance = Mock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        
        mock_info = {'title': 'Integration Test Video', 'ext': 'mp4'}
        mock_ydl_instance.extract_info.return_value = mock_info
        
        # Mock file paths to use our sample files
        mock_ydl_instance.prepare_filename.side_effect = [
            sample_video_file,  # For video
            sample_audio_file   # For audio
        ]
        
        url = "https://www.youtube.com/watch?v=test"
        
        # Patch the download_video method to return our sample files
        # Also patch the ML voice isolation to use fallback method for testing
        with patch.object(processor, 'download_video', return_value=(sample_video_file, sample_audio_file)), \
             patch.object(processor, 'isolate_voice', side_effect=processor.isolate_voice_fallback):
            result = processor.process_youtube_url(url)
        
        # Verify all expected keys are present
        expected_keys = ['video_path', 'audio_path', 'frame_path', 'voice_path']
        for key in expected_keys:
            assert key in result
            assert os.path.exists(result[key])
        
        # Verify file types
        assert result['video_path'].endswith('.mp4')
        assert result['audio_path'].endswith('.wav')
        assert result['frame_path'].endswith('.jpg')
        assert result['voice_path'].endswith('.wav')
    
    def test_main_function_argument_parsing(self):
        """Test main function argument parsing."""
        import sys
        from unittest.mock import patch
        from youtube_processor import main
        
        test_args = ['youtube_processor.py', 'https://www.youtube.com/watch?v=test', '--output-dir', 'test_output']
        
        with patch.object(sys, 'argv', test_args), \
             patch('youtube_processor.YouTubeProcessor') as mock_processor_class:
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_youtube_url.return_value = {
                'video_path': 'test.mp4',
                'audio_path': 'test.wav',
                'frame_path': 'test.jpg',
                'voice_path': 'test_voice.wav'
            }
            
            main()
            
            # Verify processor was created with correct output dir
            mock_processor_class.assert_called_once_with('test_output')
            
            # Verify processing was called with correct URL
            mock_processor.process_youtube_url.assert_called_once_with('https://www.youtube.com/watch?v=test')


if __name__ == "__main__":
    pytest.main([__file__])