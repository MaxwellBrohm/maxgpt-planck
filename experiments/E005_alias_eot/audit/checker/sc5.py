# Control-family wrong LIK picks, own classification. stdlib only.
import sys, json, collections
sys.path.insert(0,'REPO/experiments/E005_alias_eot/code')
import items_e004
D=items_e004.draw("eval")
assert 'torch' not in sys.modules
E5='REPO/experiments/E005_alias_eot/out'
E4='REPO/experiments/E004_general_updating/out'
P='HuggingFaceTB__SmolLM2-135M-Instruct'
def rt(r):
    sc=r['scores']; return all(sc['gold']>v for k,v in sc.items() if k!='gold')
for fam in ['C_noupd','C_twoslot']:
    byidx={i['idx']:i for i in D[fam]}
    for exp,d in (('E005',E5),('E004',E4)):
        c=collections.Counter()
        for s in ['s1','s2','s3','s4','s5']:
            for l in open(f'{d}/{P}__{s}__e004__plain.jsonl'):
                r=json.loads(l)
                if r['family']!=fam or rt(r): continue
                it=byidx[r['idx']]; assert it['gold']==r['cand_vals']['gold']
                sc=r['scores']; pv=r['cand_vals'][max(sc,key=sc.get)]
                st=sorted(it['stmts'],key=lambda x:x['turn'])
                asked=[x for x in st if x['obj']==it['asked']]; alast=asked[-1]['turn']
                src=[x for x in st if x['value']==pv]
                if not src: c['not_in_ctx']+=1; continue
                x=src[-1]
                if x['obj']==it['asked']: c['asked_prev']+=1
                elif x['turn']>alast: c['other_after']+=1
                else: c['other_before']+=1
                if x is st[-1]: c['=last_stmt']+=1
        print(fam,exp,dict(c),'total',sum(v for k,v in c.items() if k!='=last_stmt'))
