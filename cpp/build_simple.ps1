# Metal Defect Detection - C++ Build Script (PowerShell)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Metal Defect Detection - C++ Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "src\mdproc.cpp")) {
    Write-Host "Error: Please run this script from the cpp directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set OpenCV paths (adjust these to your installation)
$OpenCV_DIR = "C:\opencv\build\x64\vc16\lib"
$OpenCV_INCLUDE = "C:\opencv\build\include"
$OpenCV_LIBS = "C:\opencv\build\x64\vc16\lib\opencv_world4*.lib"

# Alternative paths for vcpkg installation
$VCPKG_OPENCV_INCLUDE = "C:\vcpkg\installed\x64-windows\include"
$VCPKG_OPENCV_LIBS = "C:\vcpkg\installed\x64-windows\lib\opencv_world4*.lib"

# Check which OpenCV installation exists
if (Test-Path "$OpenCV_INCLUDE\opencv2\opencv.hpp") {
    Write-Host "Found OpenCV at: $OpenCV_INCLUDE" -ForegroundColor Green
    $USE_VCPKG = $false
} elseif (Test-Path "$VCPKG_OPENCV_INCLUDE\opencv2\opencv.hpp") {
    Write-Host "Found OpenCV (vcpkg) at: $VCPKG_OPENCV_INCLUDE" -ForegroundColor Green
    $OpenCV_INCLUDE = $VCPKG_OPENCV_INCLUDE
    $OpenCV_LIBS = $VCPKG_OPENCV_LIBS
    $USE_VCPKG = $true
} else {
    Write-Host ""
    Write-Host "ERROR: OpenCV not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install OpenCV using one of these methods:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Method 1 - Download pre-built:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://opencv.org/releases/"
    Write-Host "  2. Download Windows version"
    Write-Host "  3. Extract to C:\opencv"
    Write-Host ""
    Write-Host "Method 2 - Use vcpkg:" -ForegroundColor Yellow
    Write-Host "  git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg"
    Write-Host "  cd C:\vcpkg"
    Write-Host "  .\bootstrap-vcpkg.bat"
    Write-Host "  .\vcpkg install opencv4[contrib]:x64-windows"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Create output directory
if (-not (Test-Path "bin")) {
    New-Item -ItemType Directory -Name "bin" | Out-Null
}

Write-Host ""
Write-Host "Building with OpenCV at: $OpenCV_INCLUDE" -ForegroundColor Green
Write-Host ""

# Check if cl.exe is available
try {
    $null = Get-Command cl.exe -ErrorAction Stop
    Write-Host "Found Visual Studio compiler" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Visual Studio compiler (cl.exe) not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Visual Studio Build Tools:" -ForegroundColor Yellow
    Write-Host "1. Download from https://visualstudio.microsoft.com/downloads/"
    Write-Host "2. Install 'C++ build tools' workload"
    Write-Host "3. Or run from 'Developer Command Prompt for VS'"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Build command
$buildCmd = @"
cl /EHsc /O2 /std:c++17 /I"$OpenCV_INCLUDE" /I"include" /DNDEBUG src\main.cpp src\mdproc.cpp /link "$OpenCV_LIBS" /OUT:bin\mdproc.exe
"@

Write-Host "Running build command..." -ForegroundColor Yellow
Write-Host $buildCmd -ForegroundColor Gray
Write-Host ""

# Execute build
Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Executable created: bin\mdproc.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  .\bin\mdproc.exe                    # Run with demo"
    Write-Host "  .\bin\mdproc.exe path\to\images    # Process images"
    Write-Host ""
    
    # Ask if user wants to run immediately
    $choice = Read-Host "Run the program now? (y/n)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        Write-Host ""
        Write-Host "Running mdproc..." -ForegroundColor Green
        & ".\bin\mdproc.exe"
    }
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "BUILD FAILED!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common solutions:" -ForegroundColor Yellow
    Write-Host "1. Install Visual Studio Build Tools"
    Write-Host "2. Check OpenCV installation path"
    Write-Host "3. Run from 'Developer Command Prompt for VS'"
    Write-Host "4. Make sure all dependencies are installed"
    Write-Host ""
}

Write-Host ""
Read-Host "Press Enter to exit"

