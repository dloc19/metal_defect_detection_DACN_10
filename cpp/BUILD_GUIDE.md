# C++ Build Guide for Metal Defect Detection

## Prerequisites

### 1. Install CMake
**Option A: Via Chocolatey (Recommended)**
```powershell
# Install Chocolatey if not installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install CMake
choco install cmake
```

**Option B: Manual Download**
1. Go to https://cmake.org/download/
2. Download "Windows x64 Installer"
3. Install and check "Add CMake to system PATH"

### 2. Install OpenCV
**Option A: Via vcpkg (Recommended)**
```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Integrate with Visual Studio
.\vcpkg integrate install

# Install OpenCV
.\vcpkg install opencv4[contrib]:x64-windows
```

**Option B: Pre-built OpenCV**
1. Download from https://opencv.org/releases/
2. Extract to `C:\opencv`
3. Set environment variable: `OpenCV_DIR=C:\opencv\build`

### 3. Install Visual Studio Build Tools
```powershell
# Install via Chocolatey
choco install visualstudio2022buildtools

# Or download from: https://visualstudio.microsoft.com/downloads/
# Select "C++ build tools" workload
```

## Build Instructions

### Method 1: Using CMake (Recommended)

```powershell
# Navigate to cpp directory
cd cpp

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64

# If using vcpkg OpenCV:
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# If using manual OpenCV:
cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"

# Build
cmake --build . --config Release

# Run
.\bin\Release\mdproc.exe
```

### Method 2: Simple Build Script

Create `build_simple.bat` in the `cpp` directory:

```batch
@echo off
echo Building Metal Defect Processor...

REM Set paths (adjust these to your OpenCV installation)
set OpenCV_DIR=C:\opencv\build\x64\vc16\lib
set OpenCV_INCLUDE=%OpenCV_DIR%\..\..\include
set OpenCV_LIBS=%OpenCV_DIR%\opencv_world4*.lib

REM Create output directory
if not exist "bin" mkdir bin

REM Compile
cl /EHsc /O2 /I"%OpenCV_INCLUDE%" /I"include" ^
   src\main.cpp src\mdproc.cpp ^
   /link "%OpenCV_LIBS%" ^
   /OUT:bin\mdproc.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo Run: bin\mdproc.exe
) else (
    echo Build failed!
)

pause
```

Run it:
```powershell
cd cpp
.\build_simple.bat
```

### Method 3: Using Visual Studio

1. Open Visual Studio
2. File → Open → CMake → Select `cpp/CMakeLists.txt`
3. Visual Studio will automatically configure and build
4. Set startup item to `mdproc.exe`

## Troubleshooting

### CMake not found
```powershell
# Add CMake to PATH manually
$env:PATH += ";C:\Program Files\CMake\bin"
```

### OpenCV not found
```powershell
# Check if OpenCV is installed
dir "C:\opencv\build\x64\vc16\lib\opencv_world4*.lib"

# Or with vcpkg
dir "C:\vcpkg\installed\x64-windows\lib\opencv_world4*.lib"
```

### Build errors
```powershell
# Clean build directory
rm -r build
mkdir build
cd build

# Try different generator
cmake .. -G "Visual Studio 16 2019" -A x64
```

## Testing

### Basic test
```powershell
# Create test image directory
mkdir ..\data\images

# Add some test images to ..\data\images\

# Run the program
.\bin\Release\mdproc.exe ..\data\images
```

### Performance test
```powershell
# The program will automatically benchmark itself
.\bin\Release\mdproc.exe
```

## Output

The program will show:
- Processing time per image
- Edge density analysis
- Defect detection results
- Performance benchmarks
- Summary statistics

## Next Steps

1. **Test with real images**: Add metal surface images to `../data/images/`
2. **Tune parameters**: Modify detection thresholds in the code
3. **Integrate with Python**: Use the C++ module as a backend for the Python GUI
4. **Optimize further**: Enable additional compiler optimizations

## Performance Tips

- Use Release build for maximum performance
- Enable OpenCV optimizations (automatically done)
- Use multi-threading for batch processing
- Consider GPU acceleration for large datasets
