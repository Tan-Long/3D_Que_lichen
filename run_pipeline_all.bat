@echo off
setlocal
set PY=E:\3D_Que\.venv\Scripts\python.exe
set DIR=E:\3D_Que

echo ============================================================
echo  PIPELINE FULL - 92 mau que (5 buoc)
echo  Bat dau: %date% %time%
echo ============================================================

echo.
echo [BUOC 1/4] TSDF Mesh generation...
echo ------------------------------------------------------------
%PY% "%DIR%\02_mesh_tsdf.py" --all --skip-existing
if errorlevel 1 (echo LỖING BUOC 1 >> pipeline_errors.txt)

echo.
echo [BUOC 2/4] Recolor mesh (high-res RGB)...
echo ------------------------------------------------------------
%PY% "%DIR%\03_recolor_mesh.py" --all --skip-existing
if errorlevel 1 (echo LOI BUOC 2 >> pipeline_errors.txt)

echo.
echo [BUOC 3/5] Trim mesh (cat goc/ngon)...
echo ------------------------------------------------------------
%PY% "%DIR%\04_trim_mesh.py" --all --skip-existing
if errorlevel 1 (echo LOI BUOC 3 >> pipeline_errors.txt)

echo.
echo [BUOC 4/5] Finalize mesh (lam min + cap 2 dau phang)...
echo ------------------------------------------------------------
%PY% "%DIR%\04b_finalize_mesh.py" --all --skip-existing
if errorlevel 1 (echo LOI BUOC 4 >> pipeline_errors.txt)

echo.
echo [BUOC 5/5] Detect lichen (chi dia y trang, cam/vang)...
echo ------------------------------------------------------------
%PY% "%DIR%\05_detect_lichen.py" --all --skip-existing
if errorlevel 1 (echo LOI BUOC 5 >> pipeline_errors.txt)

echo.
echo ============================================================
echo  XONG: %date% %time%
echo ============================================================

echo.
echo [PREVIEW] Khoi dong preview http://localhost:8050 ...
echo ------------------------------------------------------------
%PY% "%DIR%\04b_finalize_mesh.py" --preview-only
rem ^ kill process cu qua PID file, start moi, mo browser

pause
