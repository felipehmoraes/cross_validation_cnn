@echo off
setlocal enabledelayedexpansion

REM Caminho do Python e do script
set PYTHON_EXE=C:\Users\admin\tf-gpu\Scripts\python.exe
set SCRIPT_PATH=auto_cross_validation_cnn.py

REM Dataset fixo
set DATASET_DIR=datasets\rgb

REM Lista de resoluções (altura largura)
for %%R in (
     "32 32"
    "64 64"
    "128 128"
) do (
    for /f "tokens=1,2" %%a in ("%%~R") do (
        set ALTURA=%%a
        set LARGURA=%%b
        echo Executando com resolução !ALTURA!x!LARGURA!
        set "ALTURA=!ALTURA!"
        set "LARGURA=!LARGURA!"
        set "DATASET_DIR=%DATASET_DIR%"
        cmd /c ""%PYTHON_EXE%" "%SCRIPT_PATH%""
        echo ---------------------------------------
    )
)

echo Todos os testes foram executados.

