import json, collections
A='REPO/experiments/E005_alias_eot/al'
items=[json.loads(l) for l in open(A+'/al_items.jsonl')]
print('n items',len(items), collections.Counter(i['cell'] for i in items))
ex=[i for i in items if i['cell']=='AL2'][0]
for s in ex['stmts']: print(s)
def ncorr(i): return [s for s in i['stmts'] if s.get('ref')=='alias' or (s.get('alias') and s.get('role')!='orig')]
two=[i for i in items if len(ncorr(i))==2]
print('two-alias-correction items',len(two),collections.Counter(i['cell'] for i in two))
def title(a): return a.split()[0]
shared=[i for i in two if title(ncorr(i)[0]['alias'] or '')==title(ncorr(i)[1]['alias'] or '')]
print('shared title',len(shared),collections.Counter(i['cell'] for i in shared))
ids={i['idx'] for i in shared}
def rt(r):
    sc=r['scores']; return all(sc['gold']>v for k,v in sc.items() if k!='gold')
for m in ['e005_s1','e005_s2','e005_s3','e005_s4','e005_s5','e004_s1','e004_s2','e004_s3','e004_s4','e004_s5','base']:
    out={}
    for r in ('plain','chat'):
        L=[json.loads(l) for l in open(f'{A}/out/{m}__al__{r}.jsonl')]
        G=[json.loads(l) for l in open(f'{A}/out/{m}__gen_al__{r}.jsonl')]
        out[r]=(sum(rt(x) for x in L), sum(x['right']!=rt(x) for x in L), sum(g['strict'] for g in G),
                sum(rt(x) for x in L if x['idx'] in ids), sum(g['strict'] for g in G if g['idx'] in ids),
                {c:sum(rt(x) for x in L if x['cell']==c) for c in ['AL1','AL2','AL3','AL4']})
    p,c=out['plain'],out['chat']
    print(m,'LIKplain %d/64 %.3f mism %d cells %s | GENchat %d/64 %.3f | GENplain %d LIKchat %d | shared LIK p+c %d GEN p+c %d'%(p[0],p[0]/64,p[1],p[5],c[2],c[2]/64,p[2],c[0],p[3]+c[3],p[4]+c[4]))
print('--- aliases_all based')
twoall=[i for i in items if len(i['aliases_all'])==2]
sh=[i for i in twoall if title(i['aliases_all'][0])==title(i['aliases_all'][1])]
print('items with 2 aliases',len(twoall),collections.Counter(i['cell'] for i in twoall),'shared title',len(sh),collections.Counter(i['cell'] for i in sh))
ids2={i['idx'] for i in sh}
tot={}
for m in ['e005_s1','e005_s2','e005_s3','e005_s4','e005_s5','e004_s1','e004_s2','e004_s3','e004_s4','e004_s5']:
    for r in ('plain','chat'):
        L=[json.loads(l) for l in open(f'{A}/out/{m}__al__{r}.jsonl')]
        G=[json.loads(l) for l in open(f'{A}/out/{m}__gen_al__{r}.jsonl')]
        e=m[:4]
        tot.setdefault(e,[0,0,0,0])
        tot[e][0]+=sum(rt(x) for x in L if x['idx'] in ids2); tot[e][1]+=sum(1 for x in L if x['idx'] in ids2)
        tot[e][2]+=sum(g['strict'] for g in G if g['idx'] in ids2); tot[e][3]+=sum(1 for g in G if g['idx'] in ids2)
print('shared-title (aliases_all) LIK right/n, GEN right/n pooled seeds x renders:',tot)
# the stricter two-correction set pooled
tot={}
for m in ['e005_s1','e005_s2','e005_s3','e005_s4','e005_s5','e004_s1','e004_s2','e004_s3','e004_s4','e004_s5']:
    for r in ('plain','chat'):
        L=[json.loads(l) for l in open(f'{A}/out/{m}__al__{r}.jsonl')]
        G=[json.loads(l) for l in open(f'{A}/out/{m}__gen_al__{r}.jsonl')]
        e=m[:4]; tot.setdefault(e,[0,0,0,0])
        tot[e][0]+=sum(rt(x) for x in L if x['idx'] in ids); tot[e][1]+=sum(1 for x in L if x['idx'] in ids)
        tot[e][2]+=sum(g['strict'] for g in G if g['idx'] in ids); tot[e][3]+=sum(1 for g in G if g['idx'] in ids)
print('shared-title two-correction (AL2/AL3) pooled:',tot)
# AL3 other_obj items
o=[i for i in items if i['cell']=='AL3' and i['meta'].get('placement')=='other_obj']
print('AL3 other_obj n',len(o))
