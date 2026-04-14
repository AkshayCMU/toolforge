@echo off
REM ============================================================================
REM toolforge setup script: One-command environment + demo all-in-one
REM ============================================================================
REM Run this after cloning:
REM   cd toolforge
REM   setup.bat
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo toolforge: Offline synthetic tool-use conversation generator setup
echo ============================================================================
echo.

REM 1. Check for .env; create from example if missing
if not exist .env (
    echo [1/5] Creating .env from template...
    if exist .env.example (
        copy .env.example .env > nul
        echo        ✓ Created .env
        echo.
        echo        NEXT: Edit .env and add your ANTHROPIC_API_KEY
        echo        Then re-run this script.
        echo.
        pause
        exit /b 1
    ) else (
        echo.        ERROR: .env.example not found
        exit /b 1
    )
)

REM 2. Check for API key in .env
for /f "usebackq tokens=2 delims==" %%i in (`findstr /R "^ANTHROPIC_API_KEY" .env`) do set API_KEY=%%i
if "!API_KEY!"=="" (
    echo [1/5] ERROR: ANTHROPIC_API_KEY not set in .env
    echo        Edit .env, add your key, then re-run this script.
    echo.
    pause
    exit /b 1
)
echo [1/5] ✓ API key configured

REM 3. Activate venv + install
echo [2/5] Setting up Python virtual environment...
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [3/5] Installing toolforge...
pip install -e . --quiet
if !errorlevel! neq 0 (
    echo        ERROR: pip install failed
    exit /b 1
)
echo        ✓ Installed

REM 4. Build with mini fixture data
echo.
echo [4/5] Building tool registry and graph (mini fixture, ~30 seconds)...
toolforge build --data-dir tests\fixtures\toolbench_mini\data\toolenv\tools --examples-dir tests\fixtures\toolbench_mini\data\toolenv\tools
if !errorlevel! neq 0 (
    echo        ERROR: Build failed
    exit /b 1
)
echo        ✓ Build complete

REM 5. Quick smoke test
echo.
echo [5/5] Running smoke test (2 conversations, ~10 seconds)...
toolforge generate --n 2 --seed 42
if !errorlevel! neq 0 (
    echo        ERROR: Generate failed
    exit /b 1
)
echo        ✓ Smoke test passed

REM Done
echo.
echo ============================================================================
echo ✓ Setup complete!
echo.
echo Next steps:
echo   toolforge evaluate --in runs/dataset.jsonl --diversity
echo   toolforge generate --n 5 --seed 42  # Larger test run
echo.
echo For full diversity experiment:
echo   toolforge generate --n 120 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl
echo   toolforge generate --n 120 --seed 42 --out runs/run_b.jsonl
echo   toolforge evaluate --in runs/run_a.jsonl --diversity --out reports/run_a.json
echo   toolforge evaluate --in runs/run_b.jsonl --diversity --out reports/run_b.json
echo   toolforge compare --a reports/run_a.json --b reports/run_b.json
echo ============================================================================
echo.

endlocal
