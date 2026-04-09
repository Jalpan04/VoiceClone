@echo off
echo ==================================================
echo NATIVE XTTS FINE-TUNING SETUP (NO PINOKIO)
echo ==================================================
cd /d "d:\python projects\clone\xtts-finetune-webui"

IF NOT EXIST "venv" (
    echo [INFO] Creating Python 3.11 Virtual Environment...
    py -3.11 -m venv venv
)

echo [INFO] Activating Environment...
call venv\Scripts\activate.bat

echo [INFO] Installing CUDA PyTorch 2.1.2...
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo [INFO] Installing XTTS Requirements...
python -m pip install -r requirements.txt

echo [INFO] Starting XTTS WebUI!
python xtts_demo.py

pause
