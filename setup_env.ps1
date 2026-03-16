$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
py -m venv .venv

Write-Host "Activating virtual environment & upgrading pip..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA 12.1 support..."
& .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing data augmentation and processing libraries..."
& .\.venv\Scripts\python.exe -m pip install librosa numpy pandas matplotlib scipy soundfile

Write-Host "Environment setup complete!"
