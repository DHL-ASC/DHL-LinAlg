@echo off
SET "mypath=%~dp0%build"

cd /D %mypath%

cmake -G"MinGW Makefiles" ..

mingw32-make


