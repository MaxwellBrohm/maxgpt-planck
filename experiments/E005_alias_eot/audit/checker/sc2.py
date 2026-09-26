import json, collections
E5='REPO/experiments/E005_alias_eot'
E4='REPO/experiments/E004_general_updating'
P='HuggingFaceTB__SmolLM2-135M-Instruct'
for exp,d in (('E005',E5),('E004',E4)):
    for tag in ['base','s1','s2','s3','s4','s5']:
        R=[json.loads(l) for l in open(f'{d}/transcripts/{P}__{tag}__greedy.jsonl')]
        T=[t for r in R for t in r['turns'] if 'assistant' in t]
        cap=sum(t['flags']['hit_max'] for t in T); eos=sum(bool(t['stopped_eos']) for t in T)
        lp=sum('latest plan' in t['assistant'].lower() for t in T)
        # checks passed: count True in checks
        npass=0; ntot=0
        for r in R:
            c=r.get('checks') or {}
            vals=c.values() if isinstance(c,dict) else c
            for v in vals:
                if isinstance(v,bool): ntot+=1; npass+=v
                elif isinstance(v,dict) and 'pass' in v: ntot+=1; npass+=bool(v['pass'])
        print(exp,tag,'conv',len(R),'turns',len(T),'capped',cap,'eos',eos,'latest plan share %.2f'%(lp/len(T)),'checks',npass,'/',ntot)
# E005 eval chat stopping
for tag in ['s1','s2','s3','s4','s5']:
    G=[json.loads(l) for l in open(f'{E5}/out/{P}__{tag}__gen_e004__chat.jsonl')]
    print('E005',tag,'chat eval stop',collections.Counter(g['stop'] for g in G),'capped',sum(g['capped'] for g in G),'max n_new',max(g['n_new'] for g in G))
for tag in ['s1','s2','s3','s4','s5']:
    G=[json.loads(l) for l in open(f'{E4}/out/{P}__{tag}__gen_e004__chat.jsonl')]
    print('E004',tag,'chat eval stop',collections.Counter(g['stop'] for g in G),'capped',sum(g['capped'] for g in G))
