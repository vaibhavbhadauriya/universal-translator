# Universal Translator - Phase 1 Local MVP

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is a real-time, local-first speech translator built using Python. It captures audio from the microphone, uses Whisper for transcription, Helsinki-NLP models for translation, and gTTS for speech synthesis, all running primarily on a local GPU.

This version is the Phase 1 MVP, specifically configured for **English-to-Hindi** translation.

## Features

* **Real-time Processing:** Uses non-blocking audio capture and processing.
* **Voice Activity Detection (VAD):** Employs WebRTC VAD to detect speech segments automatically.
* **Local AI Models:** Runs AI models locally for privacy and offline capability (requires CUDA-enabled GPU).
    * [cite_start]**ASR:** OpenAI Whisper ('medium' model) for English transcription[cite: 7].
    * [cite_start]**Translation:** Helsinki-NLP ('opus-mt-en-hi') for English-to-Hindi translation[cite: 8].
    * **TTS:** Google Text-to-Speech (gTTS) for Hindi speech output.
* **GPU Acceleration:** Leverages NVIDIA GPUs via PyTorch and CUDA for faster inference.
* **Structured Logging:** Uses Loguru for clear console and file-based (JSON) logging.
* **Configuration:** Managed via `.env` file and Pydantic.

## Prerequisites

* **Python:** Version 3.10 or higher recommended.
* **NVIDIA GPU:** A CUDA-enabled GPU with at least 4GB VRAM (6GB+ recommended for the 'medium' model).
* **NVIDIA Drivers & CUDA:** Correct NVIDIA drivers and CUDA Toolkit (version 11.8 recommended for the provided setup) installed. Verify with `nvidia-smi`.
* **FFmpeg:** Required by Whisper for audio processing. [cite_start]Install via `winget install -e --id Gyan.FFmpeg` on Windows (as Admin), `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu, or `brew install ffmpeg` on macOS[cite: 7].
* **(Linux/macOS):** `portaudio` development libraries (`sudo apt-get install portaudio19-dev` on Debian/Ubuntu, `brew install portaudio` on macOS). Required for PyAudio.
* **(Linux only):** `mpg123` or similar (`sudo apt-get install mpg123`) if using the default Linux TTS playback in `ai_models.py`.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/vaibhavbhadauriya/universal-translator.git](https://github.com/vaibhavbhadauriya/universal-translator.git)
    cd universal-translator
    ```

2.  **Create and Activate Virtual Environment:**
    * It's highly recommended to use a virtual environment.
    ```bash
    # Create the environment
    python -m venv venv
    # Activate it (Windows CMD)
    venv\Scripts\activate
    # Activate it (Windows PowerShell)
    venv\Scripts\Activate.ps1
    # Activate it (macOS/Linux)
    source venv/bin/activate
    ```
    Your terminal prompt should now show `(venv)`.

3.  **Install PyTorch with CUDA:**
    * PyTorch needs to be installed first to ensure correct CUDA support for your GPU. Use the command matching your CUDA version (CUDA 11.8 is used here).
    ```bash
    # Make sure venv is active!
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Install Dependencies:**
    * [cite_start]Install all other required packages using the `requirements.txt` file[cite: 9].
    ```bash
    pip install -r requirements.txt
    ```
    * **Note on PyAudio (Windows):** If `pip install -r requirements.txt` fails with errors related to `pyaudio`, you might need to install it manually from a pre-built wheel. Download the `.whl` file matching your Python version (e.g., `cp310` for Python 3.10) from [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install it using: `pip install C:\path\to\your\downloaded\PyAudio-....whl`

5.  **Configure Environment:**
    * Create a file named `.env` in the project root.
    * Copy the following content into it. These are the default settings for the local MVP.
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
        # API Keys (leave empty for local MVP)
        ASSEMBLYAI_API_KEY=""
        ELEVENLABS_API_KEY=""
        GOOGLE_APPLICATION_CREDENTIALS=""
        JWT_SECRET_KEY=""
        ```

6.  **Verify GPU:**
    * Activate the virtual environment (`venv\Scripts\activate` or equivalent) and run:
    ```bash
    python verify_gpu.py
    ```
    You should see a success message indicating CUDA is available.

7.  **(Optional) Windows Setup Script:**
    * Alternatively, on Windows, you can try running the `setup.bat` script. Double-click it or run it from the command line: `setup.bat`. [cite_start]This script automates steps 2-4[cite: 93].

## Running the Translator

1.  **Activate Virtual Environment:**
    ```bash
    # Windows CMD
    venv\Scripts\activate
    # Windows PowerShell
    venv\Scripts\Activate.ps1
    # macOS/Linux
    source venv/bin/activate
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
