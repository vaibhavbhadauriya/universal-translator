@echo off
echo Creating fresh virtual environment...
rmdir /s /q venv 2>nul
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

echo Installing other dependencies...
python -m pip install openai-whisper transformers sentencepiece gtts pydub pyaudio webrtcvad loguru pydantic pydantic-settings

echo.
echo Setup complete! Run 'python main.py' to start.
pause
