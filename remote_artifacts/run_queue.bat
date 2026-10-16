@echo off
setlocal EnableDelayedExpansion
rem Resumable experiment queue. Safe to relaunch after any shutdown:
rem  - training steps use --resume (continue from last.pt, skip if finished)
rem  - evaluation/analysis/compare steps are skipped if their output already exists
rem All model/variant selection uses the VALIDATION split only.
cd /d E:\tem_fewshot
set PYTHONUNBUFFERED=1
set PY=E:\tem_fewshot\.venv\Scripts\python.exe
set ST=run_queue_status.txt
echo LAUNCH %date% %time% >> %ST%

call :analyze E1_siamese_supcon_densenet201

call :train E1_paper_densenet201 --model-type classifier
call :evaluate E1_paper_densenet201
call :analyze E1_paper_densenet201

call :compare E1_siamese_supcon_densenet201 feat_l2_closed E1_paper_densenet201 logits_closed cmp_siamese_feat_vs_paper
call :compare E1_siamese_supcon_densenet201 head_l2_closed E1_paper_densenet201 logits_closed cmp_siamese_head_vs_paper
call :compare E1_siamese_supcon_densenet201 feat_l2_gfsl_raw E1_paper_densenet201 feat_l2_gfsl_raw cmp_gfsl_siamese_vs_paperfeat

call :train E2_ce22_densenet201 --model-type classifier --train-classes all22 --train-source train --sampler balanced --augment --max-shift 8 --ce-head linear --optimizer adamw --lr-backbone 5e-5 --lr-head 5e-4 --weight-decay 1e-4 --schedule cosine --warmup-epochs 2 --epochs 40 --batch-size 32 --memory-efficient true --select-metric val_macro_f1 --patience 12
call :evaluate E2_ce22_densenet201 --protocols closed
call :analyze E2_ce22_densenet201

for %%S in (1 2) do (
  call :train E1_siamese_supcon_densenet201_seed%%S --model-type metric --loss supcon --select-metric val_feat_macro_f1 --seed %%S
  call :evaluate E1_siamese_supcon_densenet201_seed%%S
  call :analyze E1_siamese_supcon_densenet201_seed%%S
)

call :train E6_supcon_auxce05_densenet201 --model-type metric --loss supcon --aux-ce-weight 0.5 --select-metric val_feat_macro_f1
call :evaluate E6_supcon_auxce05_densenet201
call :analyze E6_supcon_auxce05_densenet201

call :train E6_triplet_densenet201 --model-type metric --loss triplet --select-metric val_feat_macro_f1
call :evaluate E6_triplet_densenet201
call :analyze E6_triplet_densenet201

echo DONE %date% %time% >> %ST%
goto :eof

rem ---------------------------------------------------------------- subroutines
:train
set RUN=%1
shift
set ARGS=
:collect
if "%~1"=="" goto run_train
set ARGS=!ARGS! %1
shift
goto collect
:run_train
if exist runs\%RUN%\train_summary.json (
  echo skip_train_%RUN% >> %ST%
  goto :eof
)
%PY% train.py --data-root data --out-dir runs --run-name %RUN% --resume %ARGS% >> logs_train_%RUN%.log 2>&1
echo train_%RUN%=!errorlevel! !time! >> %ST%
goto :eof

:evaluate
if exist runs\%1\eval_full\results.json (
  echo skip_eval_%1 >> %ST%
  goto :eof
)
if not exist runs\%1\train_summary.json (
  echo cannot_eval_%1_not_trained >> %ST%
  goto :eof
)
%PY% evaluate.py --data-root data --checkpoint runs\%1\best.pt --out-dir runs\%1\eval_full %2 %3 > logs_eval_%1.log 2>&1
echo eval_%1=!errorlevel! !time! >> %ST%
goto :eof

:analyze
if exist runs\%1\analysis\analysis.json (
  echo skip_analyze_%1 >> %ST%
  goto :eof
)
if not exist runs\%1\train_summary.json (
  echo cannot_analyze_%1_not_trained >> %ST%
  goto :eof
)
%PY% analyze.py --data-root data --checkpoint runs\%1\best.pt > logs_an_%1.log 2>&1
echo analyze_%1=!errorlevel! !time! >> %ST%
goto :eof

:compare
if exist runs\%5.json (
  echo skip_compare_%5 >> %ST%
  goto :eof
)
if not exist runs\%1\analysis\predictions.npz goto :eof
if not exist runs\%3\analysis\predictions.npz goto :eof
%PY% compare.py --a runs\%1\analysis\predictions.npz --a-key %2 --b runs\%3\analysis\predictions.npz --b-key %4 --out runs\%5.json > logs_%5.log 2>&1
echo compare_%5=!errorlevel! !time! >> %ST%
goto :eof
