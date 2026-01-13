$ErrorActionPreference = "Stop"

$cmake = "C:\Program Files\CMake\bin\cmake.exe"
$projectDir = "c:\Workspaces\dynamic_nn"
$buildDir = Join-Path $projectDir "build"

# Create build directory if it doesn't exist
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Set-Location $buildDir

Write-Host "=== Configuring CMake ===" -ForegroundColor Cyan
& $cmake .. -G "Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Building Project ===" -ForegroundColor Cyan
& $cmake --build . --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Build Completed Successfully ===" -ForegroundColor Green
