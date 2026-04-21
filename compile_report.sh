#!/bin/bash
# Compile LaTeX research report with embedded figures
#
# This script compiles RESEARCH_REPORT.tex into a PDF with:
# - All content from RESEARCH_DOCUMENTATION.md
# - 4 embedded PNG figures
# - Professional tables with booktabs formatting
# - Table of contents, list of figures, and list of tables
#
# Usage: ./compile_report.sh

echo "Compiling RESEARCH_REPORT.tex..."
echo "================================"

# First pass
echo "Running first LaTeX pass..."
lualatex -interaction=nonstopmode RESEARCH_REPORT.tex > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ First pass failed. Check RESEARCH_REPORT.log for errors."
    exit 1
fi

# Second pass (for TOC and cross-references)
echo "Running second LaTeX pass..."
lualatex -interaction=nonstopmode RESEARCH_REPORT.tex > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Second pass failed. Check RESEARCH_REPORT.log for errors."
    exit 1
fi

# Check if PDF was generated
if [ -f RESEARCH_REPORT.pdf ]; then
    SIZE=$(ls -lh RESEARCH_REPORT.pdf | awk '{print $5}')
    PAGES=$(pdfinfo RESEARCH_REPORT.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    echo ""
    echo "✅ Compilation successful!"
    echo "   PDF: RESEARCH_REPORT.pdf"
    echo "   Size: $SIZE"
    if [ -n "$PAGES" ]; then
        echo "   Pages: $PAGES"
    fi
    echo ""
    echo "Opening PDF..."
    open RESEARCH_REPORT.pdf
else
    echo "❌ PDF not generated. Check RESEARCH_REPORT.log for errors."
    exit 1
fi
