#!/bin/zsh
# one-line status of the E003 queue (read-only)
L=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E003_correction_floor/logs
last=$(tail -1 $L/queue.txt 2>/dev/null)
cur=$(ls -t $L/*.guard.log 2>/dev/null | head -1)
name=$(basename "$cur" .guard.log 2>/dev/null)
prog=$(grep -E "^(probe|step [0-9]+0 |scored|DONE|loaded)" $L/$name.log 2>/dev/null | tail -1 | cut -c1-110)
mem=$(tail -1 "$cur" 2>/dev/null | cut -c1-60)
echo "$(date +%T) | q: $last | job: ${name:-none} | $prog | $mem"
