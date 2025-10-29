"""
Universal Translator - Phase 1 Local MVP
Real-time speech translation using Whisper, Helsinki-NLP, and gTTS
"""

import asyncio
import pyaudio
import webrtcvad
import queue  # Standard library thread-safe queue
import numpy as np
from logger import logger
from config import settings
from ai_models import (
    run_local_whisper,
    run_local_translation,
    run_local_tts,
    load_whisper_model,
    load_translation_model
)

# --- Audio Configuration (CORRECTED) ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz (required by Whisper/AssemblyAI/WebRTC VAD)
CHUNK_DURATION_MS = 30  # 30ms frames (valid for WebRTC VAD: 10, 20, or 30ms)

# CORRECTED CHUNK_SIZE CALCULATION
SAMPLES_PER_CHUNK = int(RATE * (CHUNK_DURATION_MS / 1000.0))
CHUNK_SIZE = SAMPLES_PER_CHUNK * 2  # *2 for 16-bit audio (2 bytes per sample)
BYTES_PER_SAMPLE = 2

logger.info("=" * 60)
logger.info("Universal Translator - Phase 1 Local MVP")
logger.info("=" * 60)
logger.info(f"Audio Configuration:")
logger.info(f"  Sample Rate: {RATE} Hz")
logger.info(f"  Channels: {CHANNELS} (mono)")
logger.info(f"  Frame Duration: {CHUNK_DURATION_MS} ms")
logger.info(f"  Samples per Chunk: {SAMPLES_PER_CHUNK}")
logger.info(f"  Chunk Size: {CHUNK_SIZE} bytes")

# Validate configuration for WebRTC VAD
assert CHUNK_DURATION_MS in [10, 20, 30], f"Invalid frame duration: {CHUNK_DURATION_MS}ms (must be 10, 20, or 30)"
assert RATE in [8000, 16000, 32000, 48000], f"Invalid sample rate: {RATE}Hz (must be 8000, 16000, 32000, or 48000)"
# Recalculated assert based on corrected calculation
# assert CHUNK_SIZE == 960, f"Expected 960 bytes for 30ms@16kHz, got {CHUNK_SIZE}"

logger.success("Audio configuration validated for WebRTC VAD")

# --- Global Queues ---
audio_queue = queue.Queue(maxsize=100)
orchestra_queue = asyncio.Queue()

# --- 1. Audio Callback (Runs in PyAudio's C Thread) ---
def audio_callback(in_data, frame_count, time_info, status):
    """
    Lightweight callback that runs in PyAudio's separate thread.
    Only responsibility: put raw audio bytes into the thread-safe queue.
    NO heavy processing here to avoid blocking the audio stream.
    """
    if status:
        logger.warning(f"PyAudio callback status: {status}")
    
    try:
        # Non-blocking put - drops frame if queue is full
        audio_queue.put_nowait(in_data)
    except queue.Full:
        logger.warning("Audio queue is full, dropping frame.")
    
    return (None, pyaudio.paContinue)


# --- 2. Audio Processor (VAD & Speech Detection) ---
async def audio_processor():
    """
    Processes audio chunks from the thread-safe queue.
    Uses WebRTC VAD to detect speech segments and silence.
    """
    logger.info("Audio processor started.")
    
    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(2)
    logger.info("WebRTC VAD initialized (aggressiveness mode: 2)")
    
    current_speech = bytearray()
    is_speaking = False
    silence_frames = 0
    frames_to_wait = 25  # ~750ms at 30ms/frame
    
    loop = asyncio.get_event_loop()
    
    while True:
        try:
            # Get audio from thread-safe queue using executor
            audio_chunk = await loop.run_in_executor(
                None,  # Use default executor
                audio_queue.get,
                True,  # block=True
                0.1    # timeout=0.1 seconds
            )
            
            # Verify chunk size (critical for WebRTC VAD)
            if len(audio_chunk) != CHUNK_SIZE:
                logger.error(f"Invalid chunk size: {len(audio_chunk)} bytes (expected {CHUNK_SIZE})")
                continue
            
            # Run VAD on the chunk
            try:
                is_speech_chunk = vad.is_speech(audio_chunk, RATE)
            except Exception as vad_error:
                logger.error(f"VAD error: {vad_error} | Chunk size: {len(audio_chunk)} bytes")
                continue
            
            # Speech detection state machine
            if is_speech_chunk:
                if not is_speaking:
                    logger.info("Speech detected - recording started...")
                    is_speaking = True
                
                current_speech.extend(audio_chunk)
                silence_frames = 0
            
            else:  # Silence detected
                if is_speaking:
                    silence_frames += 1
                    current_speech.extend(audio_chunk)  # Include trailing silence
                    
                    # End of speech after sufficient silence
                    if silence_frames >= frames_to_wait:
                        duration_seconds = len(current_speech) / (RATE * 2)
                        logger.info(f"Silence detected - recording stopped")
                        logger.info(f"Captured: {len(current_speech)} bytes (~{duration_seconds:.2f}s)")
                        
                        # Only process if audio is long enough (min 0.3 seconds)
                        if len(current_speech) > RATE * 2 * 0.3:
                            await orchestra_queue.put(bytes(current_speech))
                            logger.success("Audio segment queued for processing")
                        else:
                            logger.warning(f"Audio segment too short ({duration_seconds:.2f}s), discarding")
                        
                        # Reset state
                        current_speech = bytearray()
                        is_speaking = False
                        silence_frames = 0
        
        except queue.Empty:
            await asyncio.sleep(0.01)  # Prevent busy loop
            continue
        
        except Exception as e:
            logger.error(f"Error in audio processor: {e}", exc_info=True)
            await asyncio.sleep(0.1)


# --- 3. Orchestrator (ASR -> Translation -> TTS Pipeline) ---
async def orchestrator():
    """
    Coordinates the full translation pipeline:
    1. Transcribe audio using Whisper (ASR)
    2. Translate text using Helsinki-NLP (MT)
    3. Synthesize speech using gTTS (TTS)
    """
    logger.info("Orchestrator started.")
    
    while True:
        try:
            # Get audio segment from queue
            speech_data = await orchestra_queue.get()
            logger.info(f"Processing {len(speech_data)} bytes of audio...")
            
            # 1. Automatic Speech Recognition (ASR)
            logger.info("Running Whisper transcription...")
            transcript = await run_local_whisper(speech_data)
            
            if not transcript:
                logger.warning("No transcript generated - skipping segment")
                orchestra_queue.task_done()
                continue
            
            logger.info(f"Transcript: \"{transcript}\"")
            
            # 2. Machine Translation (MT)
            logger.info("Translating text...")
            translated_text = await run_local_translation(
                transcript,
                source_lang="en",
                target_lang="hi"
            )
            
            if not translated_text:
                logger.warning("Translation failed - skipping TTS")
                orchestra_queue.task_done()
                continue
            
            logger.info(f"Translation: \"{translated_text}\"")
            
            # 3. Text-to-Speech (TTS)
            logger.info("Generating speech...")
            await run_local_tts(translated_text, lang="hi")
            
            logger.success("Pipeline complete!")
            logger.info("-" * 60)
            
            orchestra_queue.task_done()
        
        except Exception as e:
            logger.error(f"Critical error in orchestrator: {e}", exc_info=True)
            orchestra_queue.task_done()
            await asyncio.sleep(0.5)


# --- 4. Graceful Shutdown ---
async def shutdown(p, stream):
    """Gracefully shutdown and cleanup resources"""
    logger.info("Shutting down gracefully...")
    
    # Stop accepting new audio
    stream.stop_stream()
    
    logger.info("Waiting for queues to drain (max 5s)...")
    try:
        await asyncio.wait_for(
            orchestra_queue.join(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for queues to drain")
    
    # Cleanup
    stream.close()
    p.terminate()
    logger.success("Shutdown complete")


# --- 5. Main Function ---
async def main():
    logger.info("Starting main application...")
    
    # Load AI models
    load_whisper_model()
    load_translation_model()
    logger.success("All models loaded successfully")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    logger.info(f"Found {p.get_device_count()} audio devices")
    
    # Open audio stream with callback
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=SAMPLES_PER_CHUNK,  # Number of samples (not bytes)
        stream_callback=audio_callback
    )
    
    stream.start_stream()
    logger.success("=" * 60)
    logger.success("AUDIO STREAM STARTED - LISTENING FOR SPEECH...")
    logger.success("=" * 60)
    logger.info("Speak into your microphone in English")
    logger.info("Pause for 1 second after speaking to trigger processing")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Start async tasks
    processor_task = asyncio.create_task(audio_processor())
    orchestrator_task = asyncio.create_task(orchestrator())
    
    try:
        # Keep running until interrupted
        await asyncio.gather(processor_task, orchestrator_task)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C)")
        await shutdown(p, stream)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        await shutdown(p, stream)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)