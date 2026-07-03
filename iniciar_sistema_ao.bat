@echo off
echo ================================================
echo   Sistema AO - Arranque Completo
echo   Camara: Point Grey Chameleon CMLN-13S2M
echo ================================================
echo.

:: Verificar que Docker este corriendo
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker no esta corriendo. Inicia Docker Desktop primero.
    pause
    exit /b 1
)

echo [1/2] Iniciando contenedores Docker (backend, simulador, interfaz)...
cd /d c:\Sistema-AO
docker-compose up -d
if errorlevel 1 (
    echo [ERROR] Fallo al iniciar Docker.
    pause
    exit /b 1
)

echo.
echo [2/2] Iniciando daemon de camara Point Grey Chameleon (USB)...
echo       Presiona Ctrl+C para detener el daemon.
echo.
echo Interfaz disponible en: http://localhost:3000
echo Selecciona la pestana "Camara Real (Chameleon)" en la interfaz.
echo.

set PYTHONIOENCODING=utf-8
python c:\Sistema-AO\controlador\camera_daemon.py

echo.
echo [INFO] Daemon detenido.
pause
