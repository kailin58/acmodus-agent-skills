@echo off
cd /d B:\mode\pinterest
set PYTHONUSERBASE=C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages
:loop
py -3.13 -X utf8 -u dl_boards.py >> boards_dl.log 2>&1
findstr /c:"BOARD_DL_DONE" boards_dl.log >nul 2>&1 && goto done
timeout /t 30 /nobreak >nul
goto loop
:done
echo BOARD_WRAPPER_DONE >> boards_dl.log
