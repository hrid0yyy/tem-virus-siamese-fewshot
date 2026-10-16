@echo off
cd /d E:\tem_fewshot
set PYTHONUNBUFFERED=1
set PY=E:\tem_fewshot\.venv\Scripts\python.exe
echo WAITING for run_main %date% %time% > run_ce22_status.txt
:wait
findstr /c:"DONE" run_main_status.txt >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto wait
)
echo START %date% %time% >> run_ce22_status.txt
%PY% train.py --data-root data --out-dir runs --run-name E2_ce22_densenet201 --model-type classifier --train-classes all22 --train-source train --sampler balanced --augment --max-shift 8 --ce-head linear --optimizer adamw --lr-backbone 5e-5 --lr-head 5e-4 --weight-decay 1e-4 --schedule cosine --warmup-epochs 2 --epochs 40 --batch-size 32 --memory-efficient true --select-metric val_macro_f1 --patience 12 > runs_E2_ce22_train.log 2>&1
echo ce22_train_exit=%errorlevel% %time% >> run_ce22_status.txt
%PY% evaluate.py --data-root data --checkpoint runs\E2_ce22_densenet201\best.pt --protocols closed --out-dir runs\E2_ce22_densenet201\eval_full > runs_E2_ce22_eval.log 2>&1
echo ce22_eval_exit=%errorlevel% %time% >> run_ce22_status.txt
echo DONE %date% %time% >> run_ce22_status.txt
