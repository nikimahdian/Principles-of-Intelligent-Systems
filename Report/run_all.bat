@echo off
echo ========================================
echo FIS4041 Final Project - Setup and Run
echo ========================================
echo.

echo Step 1: Installing dependencies...
py -m pip install numpy pandas scikit-learn matplotlib seaborn pyswarms deap scipy
echo.

echo Step 2: Running all questions...
cd Code
py run_all_questions.py
echo.

echo ========================================
echo Execution Complete!
echo Check Results/ and Explanations/ folders
echo ========================================
pause


