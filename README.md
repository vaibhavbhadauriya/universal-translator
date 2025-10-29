# Universal Translator - Phase 1 Local MVP

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a real-time, local-first speech translator built using Python. It captures audio from the microphone, uses Whisper for transcription, Helsinki-NLP models for translation, and gTTS for speech synthesis, all running primarily on a local GPU.

This version is the Phase 1 MVP, specifically configured for **English-to-Hindi** translation.

![Flow Diagram](https://raw.githubusercontent.com/google/generative-ai-docs/main/site/en/gemini-api/docs/images/function-calling/speech-translator-flow.png) 
*(Replace this URL with a link to your own diagram if you create one)*

## Features

* **Real-time Processing:** Uses non-blocking audio capture and processing.
* **Voice Activity Detection (VAD):** Employs WebRTC VAD to detect speech segments automatically.
* **Local AI Models:** Runs AI models locally for privacy and offline capability (requires CUDA-enabled GPU).
    * **ASR:** OpenAI Whisper ('medium' model) for English transcription.
    * **Translation:** Helsinki-NLP ('opus-mt-en-hi') for English-to-Hindi translation.
    * **TTS:** Google Text-to-Speech (gTTS) for Hindi speech output.
* **GPU Acceleration:** Leverages NVIDIA GPUs via PyTorch and CUDA for faster inference.
* **Structured Logging:** Uses Loguru for clear console and file-based (JSON) logging.
* **Configuration:** Managed via `.env` file and Pydantic.

## Prerequisites

* **Python:** Version 3.10 or higher recommended.
* **NVIDIA GPU:** A CUDA-enabled GPU with at least 4GB VRAM (6GB+ recommended for the 'medium' model).
* **NVIDIA Drivers & CUDA:** Correct NVIDIA drivers and CUDA Toolkit (version 11.8 recommended for the provided setup) installed. Verify with `nvidia-smi`.
* **FFmpeg:** Required by Whisper for audio processing. (Optional for TTS if using system player).
* **(Linux/macOS):** `portaudio` development libraries (`sudo apt-get install portaudio19-dev` on Debian/Ubuntu, `brew install portaudio` on macOS).
* **(Linux only):** `mpg123` or similar (`sudo apt-get install mpg123`) if using the default Linux TTS playback in `ai_models.py`.

## Setup (Windows)

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/universal-translator.git](https://github.com/YOUR_USERNAME/universal-translator.git)
    cd universal-translator
    ```
2.  **Run Setup Script:**
    Double-click the `setup.bat` file or run it from the command line:
    ```bash
    setup.bat
    ```
    This script will:
    * Create a Python virtual environment (`venv`).
    * Activate it.
    * Upgrade pip.
    * Install PyTorch with CUDA 11.8 support.
    * Install all other required Python packages.

3.  **Configure Environment:**
    * Rename the `.env.example` file (if you create one) or create a new file named `.env`.
    * Fill in any necessary API keys if you plan to extend to Phase 2 (cloud features). For this local MVP, the defaults are mostly fine, but the file *must* exist. See `.env.example` for required fields.
        ```dotenv
        # .env
        LOG_LEVEL=INFO
        SAMPLE_RATE=16000
        CHANNELS=1
        CHUNK_DURATION_MS=30
        VAD_AGGRESSIVENESS=2
        WHISPER_MODEL=medium
        SOURCE_LANG=en
        TARGET_LANG=hi
        MAX_AUDIO_QUEUE_SIZE=100
        SILENCE_FRAMES_THRESHOLD=25
        MIN_SPEECH_DURATION_SECONDS=0.3
        ASSEMBLYAI_API_KEY="" 
        ELEVENLABS_API_KEY=""
        GOOGLE_APPLICATION_CREDENTIALS=""
        JWT_SECRET_KEY="" 
        ```

4.  **Verify GPU:**
    Activate the virtual environment (`venv\Scripts\activate`) and run:
    ```bash
    python verify_gpu.py
    ```
    You should see a success message indicating CUDA is available.

## Running the Translator

1.  **Activate Virtual Environment:**
    ```bash
    venv\Scripts\activate
    ```
2.  **Run the Main Script:**
    ```bash
    python main.py
    ```
3.  **Speak:** Once you see the "LISTENING FOR SPEECH..." message, speak clearly in **English** into your default microphone.
4.  **Pause:** Wait about 1 second after speaking.
5.  **Listen:** The application will transcribe the English audio, translate it to Hindi, and play back the Hindi audio using your system's default MP3 player.

Press `Ctrl+C` to stop the application gracefully. Logs are saved in the `logs/` directory.

## Project Structure
