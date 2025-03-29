@echo off
setlocal enabledelayedexpansion

:: Cleanup Script for Data Projects
:: Version 1.3 - Added Dev Container Cleanup

echo ############################################
echo #          Project Cleanup Utility         #
echo ############################################
echo.

:: Initialize counters
set dir_count=0
set file_count=0

echo [1/7] Cleaning cache directories...
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

echo [2/7] Removing Python cache files...
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

echo [3/7] Cleaning temporary files...
for /r %%i in (*.tmp, *.temp, ~$*) do (
    if exist "%%i" (
        del /q "%%i" 2>nul
        set /a file_count+=1
    )
)

echo [4/7] Clearing Jupyter checkpoints...
for /d /r %%i in (.ipynb_checkpoints) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
        set /a dir_count+=1
    )
)

echo [5/7] Removing .devcontainer folder...
if exist ".devcontainer" (
    rmdir /s /q ".devcontainer"
    echo Removed .devcontainer
    set /a dir_count+=1
)

echo [6/7] Cleaning .code-workspace and VS Code remote settings...
for %%f in (*.code-workspace) do (
    powershell -Command "(Get-Content '%%f') -notmatch 'remoteAuthority' | Set-Content '%%f'"
    echo Cleaned remoteAuthority in %%f
)

if exist ".vscode" (
    if not exist ".vscode\settings.json" (
        echo { > .vscode\settings.json
        echo     "remote.containers.enabled": false >> .vscode\settings.json
        echo } >> .vscode\settings.json
        echo Created VS Code settings to disable dev containers
    ) else (
        powershell -Command ^
            "$s = Get-Content .vscode\settings.json | ConvertFrom-Json;" ^
            "$s.'remote.containers.enabled' = $true;" ^
            "$s | ConvertTo-Json -Depth 10 | Set-Content .vscode\settings.json"
        echo Updated .vscode/settings.json to disable dev containers
    )
)

echo [7/7] Purging pip cache...
where pip >nul 2>&1
if %errorlevel% equ 0 (
    python -m pip cache purge 2>nul
    echo Pip cache cleared successfully
) else (
    echo Pip not found - skipping cache purge
)

echo.
echo #################################
echo #          Summary              #
echo #################################
echo Removed directories: %dir_count%
echo Deleted files: %file_count%
echo.

echo Cleanup operation completed successfully
timeout /t 3 /nobreak >nul
