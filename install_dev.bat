@echo off
echo Installing Jira AI Assistant in development mode...
pip install -e .

echo.
echo Setting PYTHONPATH environment variable...
set PYTHONPATH=%PYTHONPATH%;%CD%
setx PYTHONPATH "%PYTHONPATH%;%CD%"

echo.
echo Verifying environment configuration...
python check_env.py

echo.
echo Done! You can now run the assistant with: python main.py 