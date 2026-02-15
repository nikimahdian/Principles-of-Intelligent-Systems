@echo off
echo Compiling Persian Presentation...
echo.

REM Change to Report directory
cd /d "%~dp0"

REM Compile the full presentation
echo Compiling full presentation (presentation.tex)...
xelatex -interaction=nonstopmode presentation.tex
xelatex -interaction=nonstopmode presentation.tex

echo.
echo Compiling 7-minute version (presentation_7min.tex)...
xelatex -interaction=nonstopmode presentation_7min.tex
xelatex -interaction=nonstopmode presentation_7min.tex

echo.
echo Done! Check the PDF files:
echo - presentation.pdf (full version)
echo - presentation_7min.pdf (7-minute version)
pause

