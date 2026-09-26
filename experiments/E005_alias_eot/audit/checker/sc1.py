# Spot checks from raw records, stdlib only. No model.
import json, os, random, sys
E5='REPO/experiments/E005_alias_eot/out'
E4='REPO/experiments/E004_general_updating/out'
P='HuggingFaceTB__SmolLM2-135M-Instruct'
def load(d,tag,s,r):
    return [json.loads(l) for l in open(f'{d}/{P}__{tag}__{s}__{r}.jsonl')]
def rt(rec):
    sc=rec['scores']; g=sc['gold']
    return all(g>v for k,v in sc.items() if k!='gold')
FAM=['H1','H2','H3','H4','H5','H6','H7','C_noupd','C_twoslot']
cells={}
for exp,d in (('E005',E5),('E004',E4)):
    for tag in ['base','s1','s2','s3','s4','s5']:
        L=load(d,tag,'e004','plain'); G=load(d,tag,'gen_e004','plain')
        assert len(L)==640 and len(G)==640
        lk={f:sum(rt(r) for r in L if r['family']==f) for f in FAM}
        gn={f:sum(r['strict'] for r in G if r['family']==f) for f in FAM}
        mism=sum(rt(r)!=r['right'] for r in L)
        cells[(exp,tag)]=(lk,gn)
        fails=[f"{f}:{'L' if lk[f]<51.2 else ''}{'G' if gn[f]<51.2 else ''}" for f in FAM if lk[f]<51.2 or gn[f]<51.2]
        print(exp,tag,'LIK',[lk[f] for f in FAM],'GEN',[gn[f] for f in FAM],'right-mism',mism,'fails',fails)
        # alias
        al=[r for r in L if r['family'] in('H1','H2') and r['latest_ref']=='alias']
        ag=[r for r in G if r['family'] in('H1','H2') and r['latest_ref']=='alias']
        print('   alias n',len(al),len(ag),'LIK',sum(rt(r) for r in al),'GEN',sum(r['strict'] for r in ag))
# reading
def reading(exp,seeds):
    passes=[]; exceptH5=[]
    for s in seeds:
        lk,gn=cells[(exp,s)]
        failed={f for f in FAM if lk[f]/64<0.8 or gn[f]/64<0.8}
        passes.append(not failed); exceptH5.append(failed<= {'H5'})
    return sum(passes), sum(exceptH5)
print('E005 5 seeds pass, pass-except-H5:',reading('E005',['s1','s2','s3','s4','s5']))
print('E005 s1-s4 only:',reading('E005',['s1','s2','s3','s4']))
print('E004:',reading('E004',['s1','s2','s3','s4','s5']))
# kbig
base=load(E5,'base','kbig','plain'); b=[rt(r) for r in base]
print('base kbig',sum(b)/441)
acc={}
for exp,d in (('E005',E5),('E004',E4)):
    M=[]
    for s in ['s1','s2','s3','s4','s5']:
        K=load(d,s,'kbig','plain'); v=[rt(r) for r in K]; M.append(v)
        gained=sum(1 for x,y in zip(b,v) if y and not x); lost=sum(1 for x,y in zip(b,v) if x and not y)
        print(exp,s,'kbig acc %.4f d %.4f gained %d lost %d'%(sum(v)/441,sum(v)/441-sum(b)/441,gained,lost))
    acc[exp]=[sum(M[j][i] for j in range(5))/5 for i in range(441)]
d=[a-c for a,c in zip(acc['E005'],acc['E004'])]
md=sum(d)/441
rng=random.Random(12345); bs=[]
for _ in range(4000):
    bs.append(sum(d[rng.randrange(441)] for _ in range(441))/441)
bs.sort()
print('E005 vs E004 kbig d %.4f  CI %.4f %.4f  means %.4f %.4f'%(md,bs[99],bs[3899],sum(acc['E005'])/441,sum(acc['E004'])/441))
