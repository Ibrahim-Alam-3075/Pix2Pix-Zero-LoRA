# Pix2Pix-Zero Local Setup Script (Windows)

Write-Host "--- Initializing Pix2Pix-Zero Environment ---" -ForegroundColor Cyan

# 1. Create Virtual Environment
if (!(Test-Path "venv")) {
    Write-Host "[1/3] Creating Virtual Environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[1/3] Virtual Environment already exists." -ForegroundColor Green
}

# 2. Upgrade Pip
Write-Host "[2/3] Upgrading Pip..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Install Dependencies
Write-Host "[3/3] Installing Dependencies (this may take a few minutes)..." -ForegroundColor Yellow
.\venv\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "--- Setup Complete! ---" -ForegroundColor Cyan
Write-Host "To start using the repo, run: .\venv\Scripts\activate" -ForegroundColor Green
Write-Host "Then check the README.md for the inversion and editing commands." -ForegroundColor White
