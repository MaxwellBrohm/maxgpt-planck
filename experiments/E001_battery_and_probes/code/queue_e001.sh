#!/bin/zsh
# E001 run queue: strictly one model process at a time, each through guard.py
# (preflight: no other model job, LM Studio unloaded; waits for >= 35% free memory;
# kills on wall-clock ceiling, < 15% free memory, or > 768 MB swap growth).
# A memory or swap kill stops the whole queue; a ceiling kill or a crash moves on.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-planck/experiments/E001_battery_and_probes/code
Q=../logs/queue.txt
echo "$(date +%T) QUEUE START" >> $Q

run() {  # run <name> <ceiling_s> <args...>
  local name=$1 ceil=$2; shift 2
  $PY guard.py --name $name --ceiling $ceil -- $PY "$@"
  local rc=$?
  local killed=$(python3 -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
  echo "$(date +%T) $name exit=$rc killed=$killed" >> $Q
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* ]]; then
    echo "$(date +%T) QUEUE STOPPED: memory kill in $name" >> $Q
    exit 2
  fi
}

# 1) new control items on the previously probed models (fp32, MPS)
run bat_new_smollm2_135m_i 3600 run_battery.py HuggingFaceTB/SmolLM2-135M-Instruct --sets new
run bat_new_lfm25_350m     3600 run_battery.py LiquidAI/LFM2.5-350M --sets new
run bat_new_lfm2_350m      3600 run_battery.py LiquidAI/LFM2-350M --sets new
run bat_new_smollm2_360m_i 3600 run_battery.py HuggingFaceTB/SmolLM2-360M-Instruct --sets new
run bat_new_qwen25_05b_i   3600 run_battery.py Qwen/Qwen2.5-0.5B-Instruct --sets new
run bat_new_qwen3_06b      3600 run_battery.py Qwen/Qwen3-0.6B --sets new
run bat_new_gemma3_270m    5400 run_battery.py unsloth/gemma-3-270m-it --sets new
# 2) LFM2.5-230M: full likelihood battery + chat probe
run bat_full_lfm25_230m    3600 run_battery.py LiquidAI/LFM2.5-230M --sets new,old,khard
run chat_lfm25_230m_greedy 2700 run_chat.py LiquidAI/LFM2.5-230M --mode greedy
run chat_lfm25_230m_sampled 5400 run_chat.py LiquidAI/LFM2.5-230M --mode sampled --seeds 0 1
# 3) LFM2 size ladder: 700M, then 1.2B (700M moved to the Trash first to keep new downloads < 4 GB)
run bat_full_lfm2_700m     5400 run_battery.py LiquidAI/LFM2-700M --sets new,old,khard
if [[ -d ~/.cache/huggingface/hub/models--LiquidAI--LFM2-700M ]]; then
  mv ~/.cache/huggingface/hub/models--LiquidAI--LFM2-700M ~/.Trash/models--LiquidAI--LFM2-700M-E001 && echo "$(date +%T) moved LFM2-700M weights to the Trash" >> $Q
fi
$PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('LiquidAI/LFM2-1.2B', allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','LICENSE']))" > ../logs/download_2.log 2>&1
echo "$(date +%T) download LFM2-1.2B exit=$?" >> $Q
run bat_full_lfm2_1_2b     7200 run_battery.py LiquidAI/LFM2-1.2B --sets new,old,khard
# 4) MaxGPT-3 (Max's 235M), pretrain and SFT checkpoints, plain + its own USER:/ASSISTANT: format
run bat_full_maxgpt3_final 3600 run_battery.py maxgpt3:final --sets new,old,khard
run bat_full_maxgpt3_sft   3600 run_battery.py maxgpt3:final_sft --sets new,old,khard
# 5) Falcon-H1-Tiny-90M: chat probe on MPS (generation worked there before), battery on CPU
run chat_falcon_greedy     3600 run_chat.py tiiuae/Falcon-H1-Tiny-90M-Instruct --mode greedy
run chat_falcon_sampled    7200 run_chat.py tiiuae/Falcon-H1-Tiny-90M-Instruct --mode sampled --seeds 0 1
run bat_full_falcon_cpu   14400 run_battery.py tiiuae/Falcon-H1-Tiny-90M-Instruct --device cpu --sets new,old,khard
# 6) extras: the 2.6B reference that passes the old correction items, and the 135M base model
run bat_new_lfm2_2_6b      7200 run_battery.py LiquidAI/LFM2-2.6B --sets new --dtype bf16
run bat_new_smollm2_135m_base 3600 run_battery.py HuggingFaceTB/SmolLM2-135M --sets new
echo "$(date +%T) QUEUE DONE" >> $Q
