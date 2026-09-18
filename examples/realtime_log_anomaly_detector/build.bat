@echo off
REM Build script for chaos_anomaly_detector on Windows
REM This assumes pomaidb is already built in the parent directory

echo Building chaos_anomaly_detector...
echo Make sure pomaidb is built first in: ..\..\build

set POMAIDB_ROOT=..\..
set BUILD_DIR=%POMAIDB_ROOT%\build
set INCLUDE_DIR=%POMAIDB_ROOT%\include
set SRC_DIR=%POMAIDB_ROOT%\src
set CAPPI_DIR=%POMAIDB_ROOT%\src\capi
set UTILS_DIR=%POMAIDB_ROOT%\src\utils
set PALLOC_DIR=%POMAIDB_ROOT%\third_party\palloc\include

REM Check if build directory exists
if not exist "%BUILD_DIR%" (
    echo ERROR: pomaidb build directory not found at %BUILD_DIR%
    echo Please build pomaidb first:
    echo   cd %POMAIDB_ROOT%
    echo   mkdir build
    echo   cd build
    echo   cmake ..
    echo   cmake --build .
    exit /b 1
)

REM Try using CMake if available
where cmake >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using CMake to build...
    cmake -B build -S .
    cmake --build build --config Release
    if %ERRORLEVEL% EQU 0 (
        echo Build successful: build\Release\chaos_anomaly_detector.exe
        exit /b 0
    ) else (
        echo CMake build failed, trying manual compilation...
    )
)

REM Manual compilation with MSVC or MinGW
echo Attempting manual compilation...

REM Try MSVC cl.exe first
where cl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using MSVC compiler...
    cl.exe /EHsc /std:c++20 /I%INCLUDE_DIR% /I%SRC_DIR% /I%CAPPI_DIR% /I%UTILS_DIR% /I%PALLOC_DIR% chaos_anomaly_detector.cpp /Fe:chaos_anomaly_detector.exe /link /LIBPATH:%BUILD_DIR% pomai.lib pomai_c.lib palloc-static.lib ws2_32.lib
    if %ERRORLEVEL% EQU 0 (
        echo Build successful: chaos_anomaly_detector.exe
        exit /b 0
    )
)

REM Try MinGW g++
where g++ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using MinGW g++...
    g++ -std=c++20 -I%INCLUDE_DIR% -I%SRC_DIR% -I%CAPPI_DIR% -I%UTILS_DIR% -I%PALLOC_DIR% chaos_anomaly_detector.cpp -o chaos_anomaly_detector.exe -L%BUILD_DIR% -lpomai -lpomai_c -lpalloc-static -lws2_32
    if %ERRORLEVEL% EQU 0 (
        echo Build successful: chaos_anomaly_detector.exe
        exit /b 0
    )
)

echo ERROR: No suitable compiler found. Please install Visual Studio or MinGW.
exit /b 1
