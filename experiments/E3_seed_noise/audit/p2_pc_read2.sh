#!/bin/bash
# E3 Part 2 audit, read-only, run on the PC by pcwsl.sh: size and content of each run's score.err, the last line of
# train.out, and the SCORED mark. Home directory and user name are masked. Writes nothing.
E=~/planck/runs/E3; U=$(whoami)
for d in $E/e3_20m_*; do
  n=$(basename $d)
  echo "SCOREERR $n bytes $(stat -c %s $d/score.err) lines $(wc -l < $d/score.err)"
  head -c 2000 $d/score.err | sed -e "s#$HOME#~#g" -e "s#$U#USER#g" | sed "s/^/  ERR $n | /"
  echo "TRAINOUT_LAST $n $(tail -1 $d/train.out | sed -e "s#$HOME#~#g" -e "s#$U#USER#g")"
  echo "TRAINOUT_GREP $n resume:$(grep -c -i resum $d/train.out) nan:$(grep -c -i -E '\bnan\b|non-finite|diverg' $d/train.out) error:$(grep -c -i -E 'error|traceback' $d/train.out)"
done
