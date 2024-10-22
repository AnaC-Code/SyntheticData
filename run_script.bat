@echo off

REM Collect user input once
call input.bat

REM Set base path to your project folder
set "basePath=S:\Thesis\Code_Fixed\Report"

REM Set paths to your virtual environments and scripts using the base path
set "python312Path=%basePath%\env12\Scripts\activate.bat"
set "python38Path=%basePath%\.venv\Scripts\activate.bat"
set "ExtendedPath=%basePath%\extend_dataset.py"
set "SynthetizePath=%basePath%\synthetize_dataset.py"
set "ConnectPath=%basePath%\connect_dataset.py"
set "QualityPath=%basePath%\compare_quality.py"

REM Activate the first environment and run the first Python script
echo Activating environment env1 and running extend_dataset.py...
call %python312Path%
python %ExtendedPath%

REM Activate the second environment and run the second Python script
echo Activating environment env2 and running Synthetize_dataset.py...
call %python38Path%
python %SynthetizePath%

REM Activate the first environment and run the third Python script
echo Activating environment env1 and running connect_dataset.py...
call %python312Path%
python %ConnectPath%

REM Activate the first environment and run the fourth Python script
echo Activating environment env1 and running compare_quality.py...
call %python312Path%
python %QualityPath%

REM Deactivate the environment (if necessary)
deactivate

echo All scripts executed.
pause