@echo off
REM TNFR-GMX Sistema Optimizado Launcher
REM Lanza el engine optimizado + dashboard coordinados

echo.
echo ==========================================
echo 🚀 TNFR-GMX Sistema Optimizado
echo ==========================================
echo.
echo ⚡ Características del sistema optimizado:
echo   • CUDA DESHABILITADO (Ahorro memoria)
echo   • Cache TNFR optimizado (No GPU)
echo   • Red reducida: 7 assets principales
echo   • Límite memoria: 512MB
echo   • Frecuencia: 20s (eficiente)
echo   • Cache inteligente activado
echo.

REM Check dependencies
echo 🔍 Verificando dependencias...

python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ⚠️  Streamlit no encontrado. Instalando...
    pip install streamlit plotly psutil
    if errorlevel 1 (
        echo ❌ Error instalando dependencias
        pause
        exit /b 1
    )
)

python -c "import psutil" 2>nul
if errorlevel 1 (
    echo 💾 Instalando psutil para monitoreo memoria...
    pip install psutil
)

echo ✅ Dependencias verificadas
echo.

REM Set environment for optimized performance
set PYTHONPATH=%~dp0\src
set TNFR_CUDA_ENABLED=false
set TNFR_USE_CACHE=true
set TNFR_CACHE_ENABLED=true
set TNFR_CACHE_SECRET=gmx-tnfr-optimized-2024

REM Create directories
if not exist "data\cache" mkdir "data\cache"
if not exist "logs" mkdir "logs"

echo 🚀 Iniciando sistema optimizado...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📊 PASO 1: Lanzando Engine TNFR Optimizado
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd /d "%~dp0future-research\crypto-lab"

REM Launch optimized engine in background
start "TNFR-Engine-Optimized" /min cmd /c "python gmx_tnfr_optimized.py --mode monitor"

echo ✅ Engine optimizado iniciado en segundo plano
echo    • Límite memoria: 512MB
echo    • Assets: 7 principales (BTC, ETH, SOL, ARB, GMX, AVAX, LINK)
echo    • Cache: TNFR optimizado (no CUDA)
echo    • Frecuencia: 20s
echo.

REM Wait for engine to initialize
echo ⏳ Esperando inicialización del engine (10s)...
timeout /t 10 /nobreak >nul

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📈 PASO 2: Lanzando Dashboard Optimizado
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo 🎯 Abriendo dashboard en puerto 8502...
echo 📊 URL: http://localhost:8502
echo.
echo ⚠️  IMPORTANTE: 
echo    • Engine optimizado: Puerto automático
echo    • Dashboard: http://localhost:8502
echo    • Memoria limitada a 512MB total
echo    • Cache TNFR activo (eficiente)
echo.
echo 🛑 Presiona Ctrl+C para detener todo el sistema
echo.

REM Launch dashboard on different port to avoid conflicts
python -m streamlit run tnfr_dashboard_simple.py --server.port 8502 --server.address localhost

echo.
echo 🛑 Dashboard detenido. Cerrando engine optimizado...

REM Kill the engine process
taskkill /f /fi "WINDOWTITLE eq TNFR-Engine-Optimized*" 2>nul

echo ✅ Sistema optimizado completamente cerrado
echo.
echo 📊 Resumen de optimizaciones aplicadas:
echo   • Memoria: Reducida ~70%% vs versión CUDA
echo   • Assets: 7 principales (optimizado)
echo   • Cache: TNFR inteligente
echo   • Frecuencia: Eficiente (20s)
echo   • GPU: No utilizada (ahorro energía)
echo.
pause