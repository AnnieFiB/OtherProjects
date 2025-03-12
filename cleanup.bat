@echo off
echo Cleaning up __pycache__ directories...
for /r %%i in (__pycache__) do @if exist "%%i" rmdir /s /q "%%i"

echo Cleaning up .pyc files...
for /r %%i in (*.pyc) do @if exist "%%i" del "%%i"

echo Cleaning up .ipynb_checkpoints directories...
for /r %%i in (.ipynb_checkpoints) do @if exist "%%i" rmdir /s /q "%%i"

echo Cleaning up .pytest_cache directories...
for /r %%i in (.pytest_cache) do @if exist "%%i" rmdir /s /q "%%i"

echo Cleaning up temporary files...
for /r %%i in (*.tmp) do @if exist "%%i" del "%%i"

echo Clearing pip cache...
pip cache purge

echo Cleanup complete.
pause