# H5: gold never last statement; accuracy by introduction order. stdlib only.
import sys, json, collections
sys.path.insert(0,'REPO/experiments/E005_alias_eot/code')
import items_e004
items=items_e004.draw("eval")
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
h5=items["H5"]
print('H5 items',len(h5)); print('keys',list(h5[0].keys())); print(h5[0]['stmts'][:2])
E5='REPO/experiments/E005_alias_eot/out'
E4='REPO/experiments/E004_general_updating/out'
P='HuggingFaceTB__SmolLM2-135M-Instruct'
def rt(r):
    sc=r['scores']; return all(sc['gold']>v for k,v in sc.items() if k!='gold')
byidx={i['idx']:i for i in h5}
lastgold=0; order={}
for i in h5:
    st=sorted(i['stmts'],key=lambda s:s['turn'])
    if st[-1]['value']==i['gold']: lastgold+=1
    intro=[]
    for s in st:
        if s['obj'] not in intro: intro.append(s['obj'])
    order[i['idx']]=intro.index(i['asked'])
print('H5 items where last statement is gold:',lastgold, 'intro-order counts',collections.Counter(order.values()))
for exp,d in (('E005',E5),('E004',E4)):
    tot=collections.Counter(); ok=collections.Counter(); lastpick=0; wrong=0
    for s in ['s1','s2','s3','s4','s5']:
        L=[json.loads(l) for l in open(f'{d}/{P}__{s}__e004__plain.jsonl')]
        for r in L:
            if r['family']!='H5': continue
            it=byidx[r['idx']]; assert r['cand_vals']['gold']==it['gold']
            o=order[r['idx']]; tot[o]+=1; ok[o]+=rt(r)
            if not rt(r):
                wrong+=1
                sc=r['scores']; pick=max(sc,key=sc.get); pv=r['cand_vals'][pick]
                st=sorted(it['stmts'],key=lambda s:s['turn'])
                if st[-1]['value']==pv: lastpick+=1
    print(exp,{o:'%d/%d=%.2f'%(ok[o],tot[o],ok[o]/tot[o]) for o in sorted(tot)},'wrong',wrong,'wrong=last stmt',lastpick)
