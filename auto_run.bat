@echo off
title YSVN Vin_App Data Pipeline Auto-Runner

:: 1. Di chuyen vao thu muc du an (dung o C)
cd /d C:\YSVN_Projects\Vin_KQKD\Vin_App

echo ======================================================
echo  DANG KHOI DONG PIPELINE DU LIEU (SU DUNG VENV)
echo  Thoi gian: %date% %time%
echo ======================================================

:: 2. Kich hoat moi truong ao
if exist ".\.venv\Scripts\activate.bat" (
    echo [OK] Dang kich hoat moi truong ao .venv...
    call .\.venv\Scripts\activate.bat
) else (
    echo [LOI] Khong tim thay moi truong ao .venv tai thu muc nay!
    echo Vui long kiem tra lai duong dan: C:\YSVN_Projects\Vin_KQKD\Vin_App\.venv
    pause
    exit /b
)

:: 3. Chay script chinh
echo [RUN] Dang chay run_pipeline.py...
python run_pipeline.py

:: 4. Ket thuc
echo ======================================================
echo  PIPELINE DA KET THUC.
echo ======================================================

:: Dung lai de xem log neu chay thu cong (khong tat cua so ngay)
pause