#!/bin/zsh
# E002 queue, part 2. Replaces queue_e002.sh after its ft_135m_s1 job (the first queue's shell was
# stopped while ft_135m_s1 ran under its own guard; that job was left to finish). Same guard rules.
# Order: finish the 135M seeds, then the POST-HOC 135M checks (crossed control on the baseline, a rerun
# of seed 0 with the crossed control and saved weights, free generation, the 53-conversation chat
# probe), then the 360M block.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-planck/experiments/E002_ft_test/code
Q=../logs/queue.txt
until [ -f ../logs/ft_135m_s1.guard.json ]; do sleep 20; done
k1=$(python3 -c "import json;print(json.load(open('../logs/ft_135m_s1.guard.json')).get('killed'))" 2>/dev/null)
e1=$(python3 -c "import json;print(json.load(open('../logs/ft_135m_s1.guard.json')).get('exit'))" 2>/dev/null)
echo "$(date +%T) ft_135m_s1 exit=$e1 killed=$k1 (logged by queue B)" >> $Q
echo "$(date +%T) QUEUE B START" >> $Q
sleep 5
run() {
  local name=$1 ceil=$2; shift 2
  $PY guard.py --name $name --ceiling $ceil -- $PY -B "$@"
  local rc=$?
  local killed=$(python3 -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
  echo "$(date +%T) $name exit=$rc killed=$killed" >> $Q
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    echo "$(date +%T) QUEUE STOPPED: $killed in $name" >> $Q
    exit 2
  fi
}
M135=HuggingFaceTB/SmolLM2-135M-Instruct
M360=HuggingFaceTB/SmolLM2-360M-Instruct
S135=HuggingFaceTB__SmolLM2-135M-Instruct
S360=HuggingFaceTB__SmolLM2-360M-Instruct
W=../weights
A135=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
A360=(--bs 2 --accum 8 --max-len 768 --grad-ckpt)
# dry runs of the two new post-hoc job types (4 items / 1 conversation) before any long run of them
run dry_gen_135m 900 gen_probe.py $M135 hf --tag dry_base --limit 4
run dry_chat_135m 900 run_chat.py $M135 --mode greedy --only K_day --out ../transcripts/dry__${S135}__greedy.jsonl
for s in 2 3 4; do
  run ft_135m_s$s 5400 ft_test.py $M135 --seed $s $A135
done
# ---- post hoc, 135M
run rescore_base_135m_cross 1200 rescore.py $M135 hf --tag base --sets cross
run ft_135m_s0r 5400 ft_test.py $M135 --seed 0 --tag s0r $A135
run gen_135m_base 2400 gen_probe.py $M135 hf --tag base
for t in s0r s1 s2 s3 s4; do
  run gen_135m_$t 2400 gen_probe.py $M135 $W/${S135}__$t --tag $t
done
run chat_135m_base 1800 run_chat.py $M135 --mode greedy --out ../transcripts/${S135}__base__greedy.jsonl
for t in s0r s1 s2 s3 s4; do
  run chat_135m_$t 1800 run_chat.py $W/${S135}__$t --mode greedy --out ../transcripts/${S135}__${t}__greedy.jsonl
done
echo "$(date +%T) 135M BLOCK DONE" >> $Q
# ---- 360M
run base_360m 4800 ft_test.py $M360 --seed 0 --steps 0 $A360
for s in 0 1 2; do
  run ft_360m_s$s 10800 ft_test.py $M360 --seed $s $A360
done
run gen_360m_base 4800 gen_probe.py $M360 hf --tag base
for t in s0 s1 s2; do
  run gen_360m_$t 4800 gen_probe.py $M360 $W/${S360}__$t --tag $t
done
run chat_360m_base 3600 run_chat.py $M360 --mode greedy --out ../transcripts/${S360}__base__greedy.jsonl
for t in s0 s1 s2; do
  run chat_360m_$t 3600 run_chat.py $W/${S360}__$t --mode greedy --out ../transcripts/${S360}__${t}__greedy.jsonl
done
echo "$(date +%T) QUEUE B DONE" >> $Q
