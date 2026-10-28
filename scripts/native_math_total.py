"""Small structural OMML vocabulary for the integrated manuscript.

Only creates Word mathematics; never converts mathematical content to images.
"""
from copy import deepcopy as cp
from lxml import etree as E

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'


def e(name,**attrs):
    prefix,local=name.split(':');ns={'w':W,'m':M}[prefix];x=E.Element('{'+ns+'}'+local)
    for key,value in attrs.items():x.set('{'+ns+'}'+key,str(value))
    return x


def r(value,roman=False,size=24,script=None):
    assert str(value).strip(),repr(value)
    x=e('m:r');pr=e('m:rPr')
    if script:pr.append(e('m:scr',val=script))
    pr.append(e('m:sty',val='p' if roman else 'i'));x.append(pr)
    wp=e('w:rPr');wp.append(e('w:rFonts',ascii='Cambria Math',hAnsi='Cambria Math',eastAsia='Cambria Math'))
    wp.append(e('w:sz',val=size));wp.append(e('w:szCs',val=size));x.append(wp)
    t=e('m:t');t.text=str(value);t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');x.append(t)
    return x


def seq(*parts):
    result=[]
    for p in parts:
        if isinstance(p,(list,tuple)):result.extend(cp(x) for x in p)
        elif isinstance(p,str):result.append(r(p))
        elif p is not None:result.append(cp(p))
    return result


def slot(name,parts):
    x=e('m:'+name)
    for y in seq(parts):x.append(y)
    return x


def sub(base,index):
    x=e('m:sSub');x.append(slot('e',base));x.append(slot('sub',index));return x


def sup(base,power):
    x=e('m:sSup');x.append(slot('e',base));x.append(slot('sup',power));return x


def ss(base,index,power):
    x=e('m:sSubSup');x.append(slot('e',base));x.append(slot('sub',index));x.append(slot('sup',power));return x


def frac(num,den):
    x=e('m:f');x.append(slot('num',num));x.append(slot('den',den));return x


def delim(parts,left='(',right=')'):
    x=e('m:d');pr=e('m:dPr');pr.append(e('m:begChr',val=left));pr.append(e('m:endChr',val=right));x.append(pr);x.append(slot('e',parts));return x


def norm(parts,power=None,order=2):
    x=sub(delim(parts,'‖','‖'),r(str(order),roman=True))
    return sup(x,r(str(power),roman=True)) if power else x


def accent(base,char='̂'):
    x=e('m:acc');pr=e('m:accPr');pr.append(e('m:chr',val=char));x.append(pr);x.append(slot('e',base));return x


def bar(base):
    x=e('m:bar');pr=e('m:barPr');pr.append(e('m:pos',val='top'));x.append(pr);x.append(slot('e',base));return x


def sqrt(parts):
    x=e('m:rad');pr=e('m:radPr');pr.append(e('m:degHide',val=1));x.append(pr);x.append(e('m:deg'));x.append(slot('e',parts));return x


def summation(lower,upper,parts):
    x=e('m:nary');pr=e('m:naryPr');pr.append(e('m:chr',val='∑'));pr.append(e('m:limLoc',val='undOvr'));pr.append(e('m:grow',val=0));x.append(pr)
    x.append(slot('sub',lower));x.append(slot('sup',upper));x.append(slot('e',parts));return x


def math(*parts,size=None):
    x=e('m:oMath')
    for item in seq(*parts):x.append(item)
    if size:
        for wp in x.iter('{'+W+'}rPr'):
            for child in wp:
                if child.tag in ['{'+W+'}sz','{'+W+'}szCs']:child.set('{'+W+'}val',str(size))
    return x


def sym(name,index=None,power=None,hat=False,tilde=False,roman=False,script=None):
    x=r(name,roman=roman,script=script)
    if hat:x=accent(x)
    if tilde:x=accent(x,'̃')
    if index is not None and power is not None:return ss(x,r(index),r(power,roman=(power in ['*','c','X','Y','design','T','eval'])))
    if index is not None:x=sub(x,r(index))
    if power is not None:x=sup(x,r(power,roman=(power in ['*','c','X','Y','design','T','eval'])))
    return x


def symbols():
    d={name:r(name) for name in ['i','k','j','m','s','g','ℓ','N','J','p','q','T','V','B','b','η','β','L','t','a','F','H']}
    d.update({
      'G_g':sym('G','g'),'g(j)':seq('g',delim('j')),'n_T':sym('n','T'),'n_s':sym('n','s'),
      'A_kj':sub(r('A',script='script'),'kj'),'n_kj':sym('n','kj'),
      'x_i':sym('x','i'),'y_i':sym('y','i'),'ε_i':sym('ε','i'),'ε_ij':sym('ε','ij'),
      'x_0':sym('x','0'),'x_k':sym('x','k'),'x(t)':seq('x',delim('t')),
      'z_ijm':sym('z','ijm'),'σ_jT':sub('σ',seq('j',r(',',True),'T')),
      'ȳ_jT':sub(bar('y'),seq('j',r(',',True),'T')),
      'zbar_jmT':sub(bar('z'),seq('jm',r(',',True),'T')),
      'ȳ_jeval':sub(bar('y'),seq('j',r(',eval',True))),
      'w_g*':sym('w','g','*'),'w_all*':ss('w',r('all',True),r('*',True)),
      'wtilde_g':sym('w','g',tilde=True),'wtilde_gm':sub(accent('w','̃'),seq('g',r(',',True),'m')),
      'w_gjm':sub(accent('w','̃'),seq('g',delim('j'),r(',',True),'m')),
      'b_j':sym('b','j'),'ŷ_ij':sym('y','ij',hat=True),'ĥ_ij':sym('h','ij',hat=True),
      'F_cdf':sub(r('F',script='script'),seq('j',r(',',True),'T')),
      'R2':sup('R',r('2',True)),'R2_macro':sub(sup('R',r('2',True)),r('macro',True)),
      'MAE':r('MAE',True),'MAE_macro':sub(r('MAE',True),r('macro',True)),
      'RMSE':r('RMSE',True),'ECE':r('ECE',True),
      'ρ_kj':sym('ρ','kj'),'ρ':r('ρ'),
      'Rik_X':sym('R','ik','X'),'Rij_Y':sym('R','ij','Y'),
      'Rbar_kX':ss(bar('R'),'k',r('X',True)),'Rbar_jY':ss(bar('R'),'j',r('Y',True)),
      'X_Tc':ss('X','T',r('c',True)),'y_jTc':ss('y',seq('j',r(',',True),'T'),r('c',True)),
      'βhat_j':sym('β','j',hat=True),'λ_1':sym('λ','1'),'λ_2':sym('λ','2'),'λ_G':sym('λ','G'),
      'X_V':sym('X','V'),'X_V-k':sub('X',seq('V',r(',−',True),'k')),
      'xtilde_k':sym('x','k',tilde=True),'Δ_cond':ss('Δ','kj',r('cond',True)),
      'H_kj':sym('H','kj'),'r_ℓkj':sub('r',seq('ℓ',r(',',True),'kj')),
      'π_kj':sym('π','kj'),'S_boot':ss(r('S',script='script'),'b',delim('j')),
      'I_event':seq(r('I',True),delim(r('·',True))),
      'c_k':sym('c','k'),'τ_k':sym('τ','k'),'OAV_k':sub(r('OAV',True),'k'),
      'r_ik_s':ss('r','ik',delim('s')),'Φ':r('Φ',True),'Φinv':sup(r('Φ',True),r('−1',True)),
      'u_ik':sym('u','ik'),'u_k*':sym('u','k','*'),'x_k*':sym('x','k','*'),
      'Σ_s':sym('Σ','s'),'Σhat_s':sym('Σ','s',hat=True),'α_s':sym('α','s'),
      'I_p':sub(r('I',True),'p'),'tr':r('tr',True),'D_s':sym('D','s'),'R_s':sym('R','s'),'Q_sk':sym('Q','sk'),
      'cstar_k(t)':seq(sym('c','k','*'),delim('t')),
      'v(x)':seq('v',delim('x')),'v_k(x)':seq(sym('v','k'),delim('x')),
      'v_k(x0)':seq(sym('v','k'),delim(sym('x','0'))),
      'μ_xk':sub('μ',seq('x',r(',',True),'k')),'σ_xk':sub('σ',seq('x',r(',',True),'k')),
      'μ_design':ss('μ','s',r('design',True)),'Ω_s':sym('Ω','s'),
      'd_s(x)':seq(sym('d','s'),delim('x')),
      'Δ_RMS':seq('Δ',delim(seq('x',r(',',True),sym('x','0')))),
      'f_mjT':ss('f','mj',delim('T')),'F_j':sym('F','j'),
    })
    return d
