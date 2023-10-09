@echo off
SET "mypath=%~dp0%build"

if not exist %mypath%\ (
    mkdir %mypath%
)

cd /D %mypath%

cmake -G"MinGW Makefiles" ..

mingw32-make

cmd /k
