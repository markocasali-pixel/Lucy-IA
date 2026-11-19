@echo off
setlocal
cd /d F:\Lucy-IA
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set dt=%%I
set STAMP=%dt:~0,8%_%dt:~8,4%
set DEST=F:\Lucy-IA\Data\backups\lucy_%STAMP%

mkdir "%DEST%"
robocopy "F:\Lucy-IA\src" "%DEST%\src" /E /XD __pycache__ >nul
copy "F:\Lucy-IA\main.py" "%DEST%\" >nul
copy "F:\Lucy-IA\requirements-lock.txt" "%DEST%\" >nul
if exist "F:\Lucy-IA\Data\voices" robocopy "F:\Lucy-IA\Data\voices" "%DEST%\voices" /E >nul

echo Backup concluido com sucesso em: %DEST%
pause
