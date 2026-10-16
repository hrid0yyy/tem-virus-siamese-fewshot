@echo off
rem Rare-class improvement study on frozen features. Waits for run_queue.bat to finish (one GPU job at a time).
cd /d E:\tem_fewshot
set PYTHONUNBUFFERED=1
set PY=E:\tem_fewshot\.venv\Scripts\python.exe
echo WAITING for run_queue DONE %date% %time% > run_gfsl_status.txt
:wait
findstr /b /c:"DONE" run_queue_status.txt >nul 2>&1
if errorlevel 1 (
  %SystemRoot%\System32\timeout.exe /t 60 /nobreak >nul
  goto wait
)
if exist runs\gfsl_improve\gfsl_improve.json if not exist runs\gfsl_improve_PARTIAL_MARK.txt (
  echo skip_existing >> run_gfsl_status.txt
  goto done
)
echo START %date% %time% >> run_gfsl_status.txt
echo partial > runs\gfsl_improve_PARTIAL_MARK.txt
%PY% gfsl_improve.py --data-root data --out runs\gfsl_improve --checkpoints runs\E1_siamese_supcon_densenet201\best.pt runs\E1_siamese_supcon_densenet201_seed1\best.pt runs\E1_siamese_supcon_densenet201_seed2\best.pt runs\E6_supcon_auxce05_densenet201\best.pt runs\E1_paper_densenet201\best.pt > logs_gfsl_improve.log 2>&1
echo gfsl_improve=%errorlevel% %time% >> run_gfsl_status.txt
del runs\gfsl_improve_PARTIAL_MARK.txt
:done
%PY% collect_results.py --out report_data.json > logs_collect.log 2>&1
echo collect=%errorlevel% %time% >> run_gfsl_status.txt
echo DONE %date% %time% >> run_gfsl_status.txt
