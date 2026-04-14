# ============================================================================
# toolforge setup script: One-command environment + demo all-in-one
# ============================================================================
# Run this after cloning:
#   cd toolforge
#   .\setup.ps1
# ============================================================================

# Requires admin or unrestricted execution policy; try this if it fails:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

$ErrorActionPreference = "Stop"

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "toolforge: Offline synthetic tool-use conversation generator setup" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

# 1. Check for .env; create from example if missing
if (-not (Test-Path ".env")) {
    Write-Host "[1/5] Creating .env from template..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "       ✓ Created .env`n" -ForegroundColor Green
        Write-Host "       NEXT: Edit .env and add your ANTHROPIC_API_KEY" -ForegroundColor White
        Write-Host "       Then re-run this script.`n" -ForegroundColor White
        Read-Host "Press Enter to exit"
        exit 1
    } else {
        Write-Host "`n       ERROR: .env.example not found" -ForegroundColor Red
        exit 1
    }
}

# 2. Check for API key in .env
$envContent = Get-Content ".env"
$apiKey = ($envContent | Select-String "^ANTHROPIC_API_KEY" | ForEach-Object { $_.Line -split "=" | Select-Object -Last 1 }).Trim()
if (-not $apiKey) {
    Write-Host "[1/5] ERROR: ANTHROPIC_API_KEY not set in .env" -ForegroundColor Red
    Write-Host "       Edit .env, add your key, then re-run this script.`n" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[1/5] ✓ API key configured" -ForegroundColor Green

# 3. Activate venv + install
Write-Host "[2/5] Setting up Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .venv\Scripts\Activate.ps1

Write-Host "[3/5] Installing toolforge..." -ForegroundColor Yellow
python -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "       ERROR: pip install failed" -ForegroundColor Red
    exit 1
}
Write-Host "       ✓ Installed" -ForegroundColor Green

# 4. Build with mini fixture data
Write-Host "`n[4/5] Building tool registry and graph (mini fixture, ~30 seconds)..." -ForegroundColor Yellow
toolforge build --data-dir tests\fixtures\toolbench_mini\data\toolenv\tools --examples-dir tests\fixtures\toolbench_mini\data\toolenv\tools
if ($LASTEXITCODE -ne 0) {
    Write-Host "       ERROR: Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "       ✓ Build complete" -ForegroundColor Green

# 5. Quick smoke test
Write-Host "`n[5/5] Running smoke test (2 conversations, ~10 seconds)..." -ForegroundColor Yellow
toolforge generate --n 2 --seed 42
if ($LASTEXITCODE -ne 0) {
    Write-Host "       ERROR: Generate failed" -ForegroundColor Red
    exit 1
}
Write-Host "       ✓ Smoke test passed" -ForegroundColor Green

# Done
Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  toolforge evaluate --in runs/dataset.jsonl --diversity`n" -ForegroundColor Gray

Write-Host "For full diversity experiment:" -ForegroundColor White
Write-Host @"
  toolforge generate --n 120 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl
  toolforge generate --n 120 --seed 42 --out runs/run_b.jsonl
  toolforge evaluate --in runs/run_a.jsonl --diversity --out reports/run_a.json
  toolforge evaluate --in runs/run_b.jsonl --diversity --out reports/run_b.json
  toolforge compare --a reports/run_a.json --b reports/run_b.json
"@ -ForegroundColor Gray

Write-Host "============================================================================`n" -ForegroundColor Cyan
