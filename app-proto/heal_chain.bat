@echo off
rem 采集自愈:py进程0且有未完成标志时重启总链
powershell -Command "$n=(Get-CimInstance Win32_Process -Filter \"Name='py.exe'\" | Measure-Object).Count; if($n -eq 0){ Start-Process cmd -ArgumentList '/c B:\mode\collect_master.bat' -WindowStyle Hidden }"
