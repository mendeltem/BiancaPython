@echo off

:: Function to display help message
:show_help
echo Usage: run_bianca_sh.bat [OPTIONS]
echo.
echo This script runs BIANCA shell with specified command line arguments.
echo.
echo Options:
echo -image=value          Path to the input image in NIfTI format. Required.
echo -mni=value            Path to the MNI template image in NIfTI format. Required.
echo -masterfile=value     Path to the masterfile containing configuration for BIANCA. Required.
echo -output=value         Path to the output directory for BIANCA results. Required.
echo -h, -help             Show this help message and exit.
goto :EOF

:: Get script directory
set "BIANCA_DIR=%~dp0"

:: Log
echo Current directory: %BIANCA_DIR%

:: Check if help option is provided
if "%~1"=="" goto show_help
:: Parse command line arguments
setlocal enabledelayedexpansion
set "args=%*"
:parse_args
if not "!args!"=="" (
  set "arg=!args:~0,1!"
  if "!arg!"=="-" (
    set "opt=!args:~1!"
    set "val="
    if "!opt:~0,1!"=="-" (
      if "!opt!"=="h" goto show_help
      if "!opt!"=="help" goto show_help
      set "opt=!opt:~1!"
    )
    for /f "tokens=1,2 delims==" %%a in ("!opt!") do (
      set "key=%%a"
      set "val=%%b"
    )
    set "args=!args:*-=!"
    if "!val!"=="" (
      set "val=%~2"
      shift
    )
    set "opt=!key!^=!val!"
    call set "opts=%%opts%% -!opt!"
  ) else (
    set "args=!args:*-=!"
    goto :parse_args
  )
)


:: Check if required options are provided
if "!opts:-image=!"=="!opts!" (
echo Image path not provided. Usage: run_bianca_sh.bat -image^=.\path\to\image.nii
exit /b 1
)
if "!opts:-mni=!"=="!opts!" (
echo MNI path not provided. Usage: run_bianca_sh.bat -mni^=.\path\to\MNI.nii
exit /b 1
)
if "!opts:-masterfile=!"=="!opts!" (
echo Masterfile path not provided. Usage: run_bianca_sh.bat -masterfile^=.\path\to\masterfile.txt
exit /b 1
)
if "!opts:-output=!"=="!opts!" (
echo Output path not provided. Usage: run_bianca_sh.bat -output^=.\path\to\output
exit /b 1
)

:: Change directory to the project directory
cd /d "%BIANCA_DIR%"

:: Check requirements
python scripts\check_requirements.py requirements.txt
if %errorlevel% equ 1 (
echo Installing missing packages...
pip install -r requirements.txt
)

:: Run BIANCA shell
python -m bianca_shell !opts!

:: Pause before exit
pause





