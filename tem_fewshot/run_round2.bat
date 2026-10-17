@echo off
setlocal EnableDelayedExpansion
rem Round 2 (revision experiments). Resumable: every step is skipped if its output already exists,
rem training continues from last.pt. Safe to relaunch after a shutdown.
rem   Phase 1: rare-class study re-run in fp32 (deterministic) with per-patch predictions -> paired tests
rem   Phase 2: two more seeds of the paper-model baseline (B1), evaluated the same way
rem   Phase 3: fp32 re-evaluation of every checkpoint (one consistent set of numbers for all tables)
rem Original AMP outputs (eval_full, analysis, gfsl_improve) are left untouched.
cd /d E:\tem_fewshot
set PYTHONUNBUFFERED=1
set PY=E:\tem_fewshot\.venv\Scripts\python.exe
set ST=run_round2_status.txt
echo LAUNCH %date% %time% >> %ST%

set S0=E1_siamese_supcon_densenet201
set S1=E1_siamese_supcon_densenet201_seed1
set S2=E1_siamese_supcon_densenet201_seed2
set AUX=E6_supcon_auxce05_densenet201
set TRI=E6_triplet_densenet201
set P0=E1_paper_densenet201
set P1=E1_paper_densenet201_seed1
set P2=E1_paper_densenet201_seed2
set CE=E2_ce22_densenet201

rem ---------------------------------------------------------------- phase 1
for %%R in (%S0% %S1% %S2% %AUX% %P0%) do call :gfsl %%R
call :paired paired_gfsl_phase1 "%S0% %S1% %S2% %AUX%" "%P0%"

rem ---------------------------------------------------------------- phase 2
call :train %P1% --model-type classifier --seed 1
call :train %P2% --model-type classifier --seed 2
for %%R in (%P1% %P2%) do (
  call :gfsl %%R
  call :evaluate %%R
  call :analyze %%R
)
call :paired paired_gfsl_final "%S0% %S1% %S2% %AUX%" "%P0% %P1% %P2%"

rem ---------------------------------------------------------------- phase 3
for %%R in (%S0% %S1% %S2% %AUX% %TRI% %P0%) do (
  call :evaluate %%R
  call :analyze %%R
)
call :evaluate %CE% --protocols closed
call :analyze %CE%
call :compare %S0% feat_l2_closed %P0% logits_closed cmp_fp32_siamese_feat_vs_paper
call :compare %S0% head_l2_closed %P0% logits_closed cmp_fp32_siamese_head_vs_paper
call :compare %S0% feat_l2_gfsl_raw %P0% feat_l2_gfsl_raw cmp_fp32_gfsl_siamese_vs_paperfeat

echo DONE %date% %time% >> %ST%
goto :eof

rem ---------------------------------------------------------------- subroutines
:train
rem training is identical to round 1 (AMP); the fp32 switch applies to evaluation only
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
set TEMFS_FP32=
%PY% train.py --data-root data --out-dir runs --run-name %RUN% --resume %ARGS% >> logs_train_%RUN%.log 2>&1
echo train_%RUN%=!errorlevel! !time! >> %ST%
goto :eof

:gfsl
if exist runs\gfsl_fp32\%1\gfsl_improve.json (
  echo skip_gfsl_%1 >> %ST%
  goto :eof
)
if not exist runs\%1\train_summary.json (
  echo cannot_gfsl_%1_not_trained >> %ST%
  goto :eof
)
set TEMFS_FP32=1
%PY% gfsl_improve.py --data-root data --out runs\gfsl_fp32\%1 --checkpoints runs\%1\best.pt --kshot 1,5 --kshot-repeats 30 > logs_gfsl_fp32_%1.log 2>&1
echo gfsl_%1=!errorlevel! !time! >> %ST%
goto :eof

:paired
if exist runs\%1.json (
  echo skip_%1 >> %ST%
  goto :eof
)
set DIRS=
set SIAM=
set BASE=
for %%R in (%~2) do (
  set DIRS=!DIRS! runs\gfsl_fp32\%%R
  set SIAM=!SIAM! %%R
)
for %%R in (%~3) do (
  set DIRS=!DIRS! runs\gfsl_fp32\%%R
  set BASE=!BASE! %%R
)
%PY% paired_gfsl.py --dir !DIRS! --siamese !SIAM! --baseline !BASE! --out runs\%1.json > logs_%1.log 2>&1
echo %1=!errorlevel! !time! >> %ST%
goto :eof

:evaluate
if exist runs\%1\eval_fp32\results.json (
  echo skip_eval_%1 >> %ST%
  goto :eof
)
if not exist runs\%1\train_summary.json (
  echo cannot_eval_%1_not_trained >> %ST%
  goto :eof
)
set TEMFS_FP32=1
%PY% evaluate.py --data-root data --checkpoint runs\%1\best.pt --out-dir runs\%1\eval_fp32 %2 %3 > logs_eval_fp32_%1.log 2>&1
echo eval_%1=!errorlevel! !time! >> %ST%
goto :eof

:analyze
if exist runs\%1\analysis_fp32\analysis.json (
  echo skip_analyze_%1 >> %ST%
  goto :eof
)
if not exist runs\%1\train_summary.json (
  echo cannot_analyze_%1_not_trained >> %ST%
  goto :eof
)
set TEMFS_FP32=1
%PY% analyze.py --data-root data --checkpoint runs\%1\best.pt --out-dir runs\%1\analysis_fp32 > logs_an_fp32_%1.log 2>&1
echo analyze_%1=!errorlevel! !time! >> %ST%
goto :eof

:compare
if exist runs\%5.json (
  echo skip_compare_%5 >> %ST%
  goto :eof
)
if not exist runs\%1\analysis_fp32\predictions.npz goto :eof
if not exist runs\%3\analysis_fp32\predictions.npz goto :eof
%PY% compare.py --a runs\%1\analysis_fp32\predictions.npz --a-key %2 --b runs\%3\analysis_fp32\predictions.npz --b-key %4 --out runs\%5.json > logs_%5.log 2>&1
echo compare_%5=!errorlevel! !time! >> %ST%
goto :eof
