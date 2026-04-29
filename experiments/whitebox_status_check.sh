#!/bin/bash
# Quick status check for white-box extraction

echo "============================================"
echo "WHITE-BOX EXTRACTION STATUS"
echo "============================================"
echo ""

echo "Completed files:"
ls -lh data/whitebox_probing/*representations.json 2>/dev/null | awk '{print "  " $9 " - " $5}'

echo ""
echo "Running processes:"
ps aux | grep "whitebox_2" | grep -v grep | wc -l | awk '{print "  " $1 " extraction processes active"}'

echo ""
echo "Latest progress:"
for file in /private/tmp/claude-501/-Users-mediratta-code-AI-Researcher/39e739ed-378c-4296-b314-237cf7be580e/tasks/b*.output; do
    if [ -f "$file" ]; then
        name=$(tail -20 "$file" | grep "EXTRACTING" | tail -1 | awk '{print $3}')
        if [ ! -z "$name" ]; then
            progress=$(tail -3 "$file" | grep "Samples:" | tail -1 || echo "Downloading...")
            echo "  $name: $progress"
        fi
    fi
done
