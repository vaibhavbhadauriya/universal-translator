"""
AI Models Module
Handles loading and running Whisper, Translation, and TTS models
"""

import torch
import whisper
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from logger import logger
from gtts import gTTS
import os
import sys

# --- Global Model Cache ---
models = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "whisper": None,
    "translator": None,
    "tokenizer": None
}

# --- GPU Memory Check ---
def check_gpu_memory():
    """Verify sufficient GPU memory before loading models"""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = gpu_props.total_memory / (1024**3)
        gpu_name = gpu_props.name
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU memory available: {gpu_mem_gb:.2f} GB")
        
        # Whisper medium needs ~1.5GB, translation ~300MB
        REQUIRED_MEMORY_GB = 2.0
        if gpu_mem_gb < REQUIRED_MEMORY_GB:
            logger.warning(f"GPU has only {gpu_mem_gb:.2f}GB - may run out of memory")
            logger.warning(f"Consider using Whisper 'small' model instead")
    else:
        logger.warning("CUDA not available - will use CPU (much slower)")

# --- Model Loading Functions ---
def load_whisper_model():
    """
    Loads the Whisper model onto the GPU.
    Using 'medium' for good balance of speed and accuracy.
    """
    if models["whisper"] is None:
        check_gpu_memory()
        logger.info("Loading Whisper 'medium' model onto GPU...")
        try:
            models["whisper"] = whisper.load_model("medium", device=models["device"])
            logger.success(f"Whisper 'medium' model loaded on {models['device']}")
        except Exception as e:
            logger.critical(f"Failed to load Whisper model: {e}")
            raise

def load_translation_model(source_lang="en", target_lang="hi"):
    """
    Loads the Helsinki-NLP translation model (English to Hindi).
    """
    if models["translator"] is None:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        logger.info(f"Loading translation model: {model_name}...")
        try:
            models["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            models["translator"] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            ).to(models["device"])
            logger.success(f"Translation model loaded on {models['device']}")
        except Exception as e:
            logger.critical(f"Failed to load translation model: {e}")
            raise

# --- Inference Functions ---
async def run_local_whisper(speech_data: bytearray) -> str:
    """
    Transcribes raw audio bytes using Whisper.
    
    Args:
        speech_data: Raw 16-bit PCM audio at 16kHz, mono
        
    Returns:
        Transcribed text, or empty string if transcription failed
    """
    if models["whisper"] is None:
        load_whisper_model()
    
    try:
        # Validate minimum audio length (at least 0.1s = 1600 samples)
        MIN_SAMPLES = int(0.1 * 16000)
        if len(speech_data) < MIN_SAMPLES * 2:  # *2 for 16-bit
            logger.warning(f"Audio too short: {len(speech_data)} bytes")
            return ""
        
        # Convert bytes to numpy array (16-bit PCM)
        audio_np = np.frombuffer(speech_data, dtype=np.int16)
        
        # Normalize to float32 in range [-1.0, 1.0]
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        logger.debug(f"Transcribing {len(audio_float)} samples (~{len(audio_float)/16000:.2f}s)")
        
        # Run Whisper transcription
        result = models["whisper"].transcribe(
            audio_float,
            fp16=torch.cuda.is_available(),  # Use FP16 on GPU for speed
            language="en",  # Specify language for 30-40% faster processing
            task="transcribe"
        )
        
        transcript = result.get("text", "").strip()
        
        if not transcript:
            logger.warning("Whisper produced no text")
        
        return transcript
        
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}")
        return ""

async def run_local_translation(text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
    """
    Translates text using Helsinki-NLP model.
    
    Args:
        text: Text to translate
        source_lang: Source language code (default: "en")
        target_lang: Target language code (default: "hi")
        
    Returns:
        Translated text, or empty string if translation failed
    """
    if models["translator"] is None:
        load_translation_model(source_lang, target_lang)
    
    try:
        # Tokenize input text
        inputs = models["tokenizer"](text, return_tensors="pt", padding=True).to(models["device"])
        
        # Generate translation
        translated = models["translator"].generate(**inputs)
        
        # Decode output
        translated_text = models["tokenizer"].decode(translated[0], skip_special_tokens=True)
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return ""

async def run_local_tts(text: str, lang: str = "hi") -> None:
    """
    Generates and plays back speech from text using gTTS.
    
    Args:
        text: Text to synthesize
        lang: Language code for TTS (default: "hi" for Hindi)
    """
    try:
        # Generate TTS
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to temporary file
        output_file = "output.mp3"
        tts.save(output_file)
        logger.debug(f"TTS audio saved to {output_file}")
        
        # Play audio using system default player
        if os.name == 'nt':  # Windows
            os.system(f'start {output_file}')
        elif sys.platform == 'darwin':  # macOS
            os.system(f'afplay {output_file}')
        else:  # Linux
            os.system(f'mpg123 {output_file}')
        
        logger.success("TTS playback complete")
        
    except Exception as e:
        logger.error(f"Error during TTS: {e}")