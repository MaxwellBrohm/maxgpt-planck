#!/bin/zsh
# one-line status of the E002 queue (read-only)
L=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E002_ft_test/logs
last=$(tail -1 $L/queue.txt)
cur=$(ls -t $L/*.guard.log 2>/dev/null | head -1)
name=$(basename $cur .guard.log)
prog=$(grep -E "^(probe|step [0-9]+0 |scored|DONE|plain:|chat:|\[)" $L/$name.log 2>/dev/null | tail -1 | cut -c1-110)
mem=$(tail -1 $cur | cut -c1-60)
echo "$(date +%T) | q: $last | job: $name | $prog | $mem"
