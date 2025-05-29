@echo off
echo ############################################
echo #       Project Cleanup Utility (Windows)  #
echo ############################################

setlocal enabledelayedexpansion

echo [1/6] Removing cache and checkpoint directories...
for %%d in (__pycache__ .ipynb_checkpoints .pytest_cache *.egg-info .mypy_cache .dvc) do (
    for /d /r %%i in (%%d) do (
        rd /s /q "%%i" 2>nul && echo   ✔ Removed directory: %%i
    )
)

echo [2/6] Removing Python cache files (.pyc, .pyo, .pyd)...
for %%f in (*.pyc *.pyo *.pyd) do (
    del /s /q "%%f" 2>nul && echo   ✔ Removed: %%f
)

echo [3/6] Removing temporary and backup files (.tmp, *~, .bak)...
for %%f in (*.tmp *~ *.bak) do (
    del /s /q "%%f" 2>nul && echo   ✔ Removed: %%f
)

echo [4/6] Removing data outputs and logs (.tsv, .log, .xlsx)...
for %%f in (*.tsv *.log *.json *.xlsx) do (
    del /s /q "%%f" 2>nul && echo   ✔ Removed: %%f
)

echo [5/6] Removing ML model files (.pkl, .npy, .npz, .joblib, .h5, .ckpt)...
for %%f in (*.pkl *.npy *.npz *.joblib *.h5 *.ckpt) do (
    del /s /q "%%f" 2>nul && echo   ✔ Removed: %%f
)

echo [6/6] Clearing pip cache...
where pip >nul 2>&1
if %errorlevel%==0 (
    pip cache purge
    echo   ✔ Pip cache cleared
) else (
    echo   ⚠️ pip not found — skipping cache purge
)

echo.
echo ✅ Cleanup complete! CSV files have been preserved.

endlocal
pause
