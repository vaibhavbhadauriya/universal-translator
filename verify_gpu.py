import torch
from logger import logger

def check_gpu():
    if torch.cuda.is_available():
        logger.success("GPU (CUDA) is available!")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch Version: {torch.__version__}")
    else:
        logger.error("GPU (CUDA) is NOT available. PyTorch will run on CPU.")
        logger.warning("Local AI models (Whisper, NLLB) will be very slow.")

if __name__ == "__main__":
    check_gpu()