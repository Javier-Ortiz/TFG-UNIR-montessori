:: #####################################
:: --- PROYECTO PARA TFG DE LA UNIR ---
:: Autor: Francisco Javier Ortiz Gonzalez
:: Fecha: Diciembre, 2025
:: Licencia: AGPL_v3
:: ######################################

@echo off
setlocal

:: =========================================================
::  VARIABLES DE CONFIGURACIÓN INICIALES
:: =========================================================
set DEST_PATH=D:\Javi\Varios\UNIR\TFG
set ENV_NAME=env_tfg
set FULL_ENV_PATH=%DEST_PATH%\%ENV_NAME%

echo.
echo =========================================================
echo  INICIANDO CONFIGURACIÓN DE ENTORNO VIRTUAL
echo =========================================================
echo  Ruta de destino: %DEST_PATH%
echo  Nombre del entorno virtual: %ENV_NAME%
echo.

:: 1. CREAR LA CARPETA DE DESTINO SI NO EXISTE
if not exist "%DEST_PATH%" (
    echo Creando directorio: %DEST_PATH%
    mkdir "%DEST_PATH%"
)

:: 2. CREA EL ENTORNO VIRTUAL
echo Creando entorno en: %FULL_ENV_PATH%
python -m venv "%FULL_ENV_PATH%"

IF %errorlevel% NEQ 0 (
    echo.
    echo ERROR CRITICO: Fallo la creación del entorno virtual.
    pause
    exit /b 1
)

echo.
echo Entorno virtual %ENV_NAME% creado con exito.
    
:: 3. INSTALA YOLO Y DEPENDENCIAS
echo =========================================================
echo INSTALANDO ULTRALYTICS(YOLO) y DEPENDENCIAS
echo =========================================================
    
:: Se usa 'pushd' para cambiar el directorio temporalmente al destino
pushd "%DEST_PATH%"
    
:: Activar entorno virtual
cd "%ENV_NAME%\Scripts"
call activate
cd ..\..
    
:: Instalando dependencias
pip install "ultralytics[track]"
pip install opencv-python
pip install numpy
    
:: 4. LIBERANDO ENTORNO VIRTUAL
call deactivate
popd

echo.
echo =========================================================
echo CONFIGURACION FINALIZADA CON EXITO.
echo El entorno %ENV_NAME% esta listo en %DEST_PATH%
echo =========================================================

pause
endlocal