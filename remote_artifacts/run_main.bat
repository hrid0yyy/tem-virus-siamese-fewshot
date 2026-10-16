@echo off
cd /d E:\tem_fewshot
set PYTHONUNBUFFERED=1
set PY=E:\tem_fewshot\.venv\Scripts\python.exe
echo START %date% %time% > run_main_status.txt
%PY% train.py --data-root data --out-dir runs --run-name E1_siamese_supcon_densenet201 --model-type metric --loss supcon --select-metric val_feat_macro_f1 > runs_E1_siamese_train.log 2>&1
echo siamese_train_exit=%errorlevel% %time% >> run_main_status.txt
%PY% evaluate.py --data-root data --checkpoint runs\E1_siamese_supcon_densenet201\best.pt --out-dir runs\E1_siamese_supcon_densenet201\eval_full > runs_E1_siamese_eval.log 2>&1
echo siamese_eval_exit=%errorlevel% %time% >> run_main_status.txt
%PY% train.py --data-root data --out-dir runs --run-name E1_paper_densenet201 --model-type classifier > runs_E1_paper_train.log 2>&1
echo paper_train_exit=%errorlevel% %time% >> run_main_status.txt
%PY% evaluate.py --data-root data --checkpoint runs\E1_paper_densenet201\best.pt --out-dir runs\E1_paper_densenet201\eval_full > runs_E1_paper_eval.log 2>&1
echo paper_eval_exit=%errorlevel% %time% >> run_main_status.txt
echo DONE %date% %time% >> run_main_status.txt
