#!/bin/zsh
# E003 audit A6: re-run the UNCHANGED analyzer in a scratch copy and byte-compare its outputs with the stored ones.
# The scratch tree holds a copy of code/, a symlink to out/ (read only), copies of the lr_pick files and a link to
# E002's results.json (the analyzer reads it for the reference line). Nothing in the experiment folder is written.
# usage: zsh a6_rerun_analyzer.sh <scratch dir>
S=${1:?scratch dir}; E=${0:A:h:h:h}; X=$E/E003_correction_floor
rm -rf $S && mkdir -p $S/E003/logs $S/E002_ft_test || exit 1
cp -R $X/code $S/E003/code && ln -s $X/out $S/E003/out && cp $X/logs/lr_pick_*.json $S/E003/logs/
ln -s $E/E002_ft_test/results.json $S/E002_ft_test/results.json
cd $S/E003/code && ~/.venvs/planck/bin/python -B -c "
import sys, runpy
sys.argv = ['analyze_e003.py']
runpy.run_path('analyze_e003.py', run_name='__main__')
assert 'torch' not in sys.modules
" > $S/stdout.txt || exit 1
cmp $S/E003/results.json $X/results.json && echo "results.json identical"
cmp $S/E003/logs/tables.txt $X/logs/tables.txt && echo "tables.txt identical"
cmp $S/stdout.txt $X/logs/analyze_stdout.txt && echo "analyze_stdout.txt identical"
