@echo off
echo ========================================
echo Metal Defect Detection - C++ Build
echo ========================================

REM Check if we're in the right directory
if not exist "src\mdproc.cpp" (
    echo Error: Please run this script from the cpp directory
    pause
    exit /b 1
)

REM Set OpenCV paths (adjust these to your installation)
set "OpenCV_DIR=C:\opencv\build\x64\vc16\lib"
set "OpenCV_INCLUDE=C:\opencv\build\include"
set "OpenCV_LIBS=C:\opencv\build\x64\vc16\lib\opencv_world4*.lib"

REM Check if OpenCV exists
if not exist "%OpenCV_INCLUDE%\opencv2\opencv.hpp" (
    echo.
    echo ERROR: OpenCV not found at %OpenCV_INCLUDE%
    echo.
    echo Please install OpenCV:
    echo 1. Download from https://opencv.org/releases/
    echo 2. Extract to C:\opencv
    echo 3. Or use vcpkg: vcpkg install opencv4[contrib]:x64-windows
    echo.
    pause
    exit /b 1
)

REM Create output directory
if not exist "bin" mkdir bin

echo.
echo Building with OpenCV at: %OpenCV_INCLUDE%
echo.

REM Compile with optimizations
cl /EHsc /O2 /std:c++17 ^
   /I"%OpenCV_INCLUDE%" ^
   /I"include" ^
   /DNDEBUG ^
   src\main.cpp src\mdproc.cpp ^
   /link "%OpenCV_LIBS%" ^
   /OUT:bin\mdproc.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable created: bin\mdproc.exe
    echo.
    echo To run:
    echo   bin\mdproc.exe
    echo.
    echo To process images:
    echo   bin\mdproc.exe path\to\images
    echo.
    
    REM Ask if user wants to run immediately
    set /p choice="Run the program now? (y/n): "
    if /i "%choice%"=="y" (
        echo.
        echo Running mdproc...
        bin\mdproc.exe
    )
) else (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Common solutions:
    echo 1. Install Visual Studio Build Tools
    echo 2. Check OpenCV installation path
    echo 3. Run from Visual Studio Developer Command Prompt
    echo.
)

echo.
pause

