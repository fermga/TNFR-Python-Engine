"""Exact algebra controls for the declared admission; no flow or root solver."""
from fractions import Fraction as Q
from pathlib import Path
from tnfr.research.artifact_io import file_receipt, write_json_once
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
def edge(a,b):return tuple(sorted((a,b)))
def rings():return {edge(o+j,o+(j+1)%9) for o in (0,9) for j in range(9)}
old_edges=rings()|{(0,9),(1,10)}
new_edges=rings()|{(4,13),(5,14)}
p=tuple(9*(i//9)+(i%9+4)%9 for i in range(18))
check('cyclic support transport',{edge(p[a],p[b]) for a,b in old_edges}==new_edges)
degree=lambda es:tuple(sum(i in e for e in es) for i in range(18))
d=degree(new_edges)
check('live degrees and total mass',sum(d)==40 and all(d[i]==(3 if i in (4,5,13,14) else 2) for i in range(18)))
lap=lambda es:tuple(tuple(sum(i in e for e in es) if i==j else -int(edge(i,j) in es) for j in range(18)) for i in range(18))
lm,lp=lap(rings()),lap(new_edges)
def entry_delta(i,j):
    return sum((int(i==a)-int(i==b))*(int(j==a)-int(j==b)) for a,b in ((4,13),(5,14)))
check('two independent edge storage quadratics',all(lp[i][j]-lm[i][j]==entry_delta(i,j) for i in range(18) for j in range(18)))
check('old one-edge support is a failing control',lap(rings()|{(4,13)})!=lp)
check('row normalization preserves mass charges',all(sum(lp[i][j] for i in range(18))==0 for j in range(18)))
check('both winding ramp lift corrections',all(k*(p[i]-4)+9*k*int(i>=5)==k*i for k in (1,2) for i in range(9)))
nom_alpha=tuple(2*(j-4) for j in range(9))+tuple(j-4 for j in range(9))
nom_phi=(0,)*9+(1,)*9
check('actual nominal joined weighted means',sum(d[i]*nom_alpha[i] for i in range(18))==3 and sum(d[i]*nom_phi[i] for i in range(18))==20)
check('weighted mean shift after recutting lifts',sum(2*9*k for k in (1,2) for i in range(5,9))==216)
for offset in (0,9):
    for j in range(8):
        v=[Q(0)]*18;v[offset+j]=Q(1);v[offset+8]=Q(-1)
        check(f'zero-sum residual weighted mean {offset}:{j}',sum(d[i]*v[i] for i in range(18))==sum(v[i] for i in (4,5,13,14)))
check('two distinct nominal contact gaps',[(nom_alpha[r]-nom_alpha[s],nom_phi[r]-nom_phi[s]) for s,r in ((4,13),(5,14))]==[(0,1),(-1,1)])
e=Q(1,10**32);phi=Q(1,1000)
check('first contact positive and bounded',0<phi-2*e and phi+2*e<Q(1,500))
check('second gap admitted on fixed negative branch',Q(2,3)<Q(25,36)-phi-2*e and Q(44,63)-phi+2*e<Q(7,10))
check('all source internal edges acute',Q(25,144)>2*e)
lower=Q(2,9)-Q(2401,240000)
upper=Q(49,200)+Q(1,500000)+2*e**2
check('independent two-contact work interval',Q(1,5)<lower<upper<Q(1,4))
check('old work allowance excluded',Q(1,10**6)<lower)
check('retained exact root bracket comparison only',Q(6573559493,9*2**32)<Q(7,40))
check('strict pi bound yields collar margin',Q(17*3,180)-Q(1,12)==Q(1,5) and 2*e<Q(1,5))
R=Q(1,12);B=Q(1,648000)
check('full-state boundary normalization',R**2/Q(4500)==B)
check('actual disconnected control trapping',8*e**2<B and 8*e**2<R**2)
q=tuple(Q(int(i==8)-int(i==0)) for i in range(18))
check('interior dipole marks and mean',p[4]==8 and p[5]==0 and d[8]==d[0]==2 and sum(d[i]*q[i] for i in range(18))==0)
check('same three affected internal edges',[q[v]-q[u] for u,v in ((7,8),(8,0),(0,1))]==[1,-2,1])
for es in (rings(),new_edges):
    check('dipole exact work quadratic on both supports',sum((q[v]-q[u])**2 for u,v in es)==6)
check('dipole degree and dual norms',sum(d[i]*q[i]**2 for i in range(18))==4 and sum(q[i]**2/d[i] for i in range(18))==1)
lam=Q(1,90);cap=Q(2);eta_l=Q(1,11000000);eta_u=Q(1,9000000)
mu=eta_l*Q(1,25)*lam**2;mw=eta_u*cap**2;eps_l=lam/4
amin=mu/2+lam**2/16;aplus=mw/2+eps_l*cap/2+eps_l**2
kappa=min(4*(lam-eps_l)/3,eps_l*mu/aplus)
v0=Q(3,4)*eta_u*cap*R**2+(aplus/lam)*R**2
check('reused conditional warmup constants',kappa==Q(1,2233865700000) and 128/kappa==285934809600000)
check('reused complete-state warmup caps',4*v0*Q(1,2**128)/(eta_l*lam)<Q(1,2**80) and cap*v0*Q(1,2**128)/amin<Q(1,2**80))
a=Q(1,2**12);h=Q(1,2**10);X=Y=Q(1,2**40);delta=Q(1,2**50);g=Q(1,3069)
ch=2*g*h;qmax=(X+ch*(Y+2*a))/(1-ch**2)
err=2*h*X+4*g*a*h**2+2*g*h*Y+4*g**2*h**2*qmax
signal_lower=5*a*h/Q(64*3216)
check('reused conditional finite recorded margin',signal_lower-2*err-4*delta>Q(1,2**38))
check('separate phase-probe work allowance',0<3*a*a/25-4*a*Y and 3*a*a+4*a*Y<Q(1,2**21))
check('phase-probe retains full trapping',X*X+(Y+2*a)**2<R*R and X*X+Y*Y+3*a*a+4*a*Y<B)
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
write_json_once(OUT/'algebra-checks.json',{'passed':True,'checks':checks,
    'source':receipt(Path(__file__)),'protocol':receipt(OUT/'scope-and-protocol.md'),
    'review':receipt(OUT/'algebra-review.md'),
    'actual_source_transits':0,'root_solves':0,'response_assessments':0,'historical_producer_replays':0,
    'boundary':'Exact finite algebra verifies graph transport, work inequalities, absence and the arithmetic of existing conditional response theorems. It does not certify actual-source entry or evaluate a new response.'})
print({'passed':True,'exact_algebra_checks':len(checks),'actual_source_transits':0})
