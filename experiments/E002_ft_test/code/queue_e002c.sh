#!/bin/zsh
# E002 follow-up: the post-hoc 53-conversation chat probe on the 135M base and fine-tuned seeds.
# The first attempt (queue_e002b.sh) was stopped by the guard (swap grew 922 MB in chat_135m_base).
# Waits for the E003 queue to finish, then runs one model process at a time under guard.py.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-planck/experiments/E002_ft_test/code
Q=../logs/queue.txt
E3=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E003_correction_floor/logs/queue.txt
echo "$(date +%T) QUEUE C WAITING for E003" >> $Q
until grep -qE "QUEUE E003 DONE|QUEUE STOPPED" $E3 && ! pgrep -f 'queue_e003\.sh' >/dev/null; do sleep 60; done
sleep 60
echo "$(date +%T) QUEUE C START" >> $Q
run() {
  local name=$1 ceil=$2; shift 2
  $PY guard.py --name $name --ceiling $ceil -- $PY -B "$@"
  local rc=$?
  local killed=$(python3 -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
  echo "$(date +%T) $name exit=$rc killed=$killed" >> $Q
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    echo "$(date +%T) QUEUE C STOPPED: $killed in $name" >> $Q; exit 2
  fi
}
M135=HuggingFaceTB/SmolLM2-135M-Instruct
S135=HuggingFaceTB__SmolLM2-135M-Instruct
W=../weights
run chat_135m_base_c 1800 run_chat.py $M135 --mode greedy --out ../transcripts/${S135}__base__greedy.jsonl
for t in s0r s1 s2 s3 s4; do
  run chat_135m_${t}_c 1800 run_chat.py $W/${S135}__$t --mode greedy --out ../transcripts/${S135}__${t}__greedy.jsonl
done
echo "$(date +%T) QUEUE C DONE" >> $Q
