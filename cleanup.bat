@echo off
setlocal enabledelayedexpansion

:: Cleanup Script for Data Projects
:: Version 1.2 - Added error handling and progress tracking

echo ############################################
echo #          Project Cleanup Utility         #
echo ############################################
echo.

:: Initialize counters
set dir_count=0
set file_count=0

echo [1/6] Cleaning cache directories...
for /r %%i in (__pycache__, .ipynb_checkpoints, .pytest_cache) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
        if !errorlevel! equ 0 (
            set /a dir_count+=1
        ) else (
            echo Failed to remove: %%i
        )
    )
)

echo [2/6] Removing Python cache files...
for /r %%i in (*.pyc, *.pyo, *.pyd) do (
    if exist "%%i" (
        del /q "%%i" 2>nul
        if !errorlevel! equ 0 (
            set /a file_count+=1
        ) else (
            echo Failed to delete: %%i
        )
    )
)

echo [3/6] Cleaning temporary files...
for /r %%i in (*.tmp, *.temp, ~$*) do (
    if exist "%%i" (
        del /q "%%i" 2>nul
        set /a file_count+=1
    )
)

echo [4/6] Clearing Jupyter checkpoints...
for /d /r %%i in (.ipynb_checkpoints) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
        set /a dir_count+=1
    )
)

echo [5/6] Purging pip cache...
where pip >nul 2>&1
if %errorlevel% equ 0 (
    python -m pip cache purge 2>nul
    echo Pip cache cleared successfully
) else (
    echo Pip not found - skipping cache purge
)

echo [6/6] Finalizing cleanup...
echo.
echo #################################
echo #          Summary              #
echo #################################
echo Removed directories: %dir_count%
echo Deleted files: %file_count%
echo.

echo Cleanup operation completed successfully
timeout /t 3 /nobreak >nul