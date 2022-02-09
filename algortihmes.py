import numpy as np

def statistique(X,Y,M) :
    X=np.array(X)
    Y=np.array(Y)
    M=np.array(M)
    M=M/np.sum(M)
    n=len(M)
    m=len(M[0])
    delta=0
    Fx=[ sum(M[i]) for i in range(n)]
    Fy=[ sum([M[i][j] for i in range(n)]) for j in range(m) ]
    for i in range(n) :
        for j in range(m) :
            delta=delta+((M[i][j]-Fx[i]*Fy[j])**2)/(Fx[i]*Fy[j])
    Contingence=(delta)/(min(n,m)-1)
    for i in range(n) :
        for j in range(m) :
            delta=delta+(M[i][j]-Fx[i]*Fy[j])**2
    Pearson=np.sqrt(delta/(1+delta)) 
    margX="les valeurs de X : {} \n les frequence de X : {}".format(X,Fx)
    margY="les valeurs de Y : {} \n les frequence de Y : {}".format(Y,Fy)
    E_glob_X=np.sum(X*Fx)
    E_glob_Y=np.sum(Y*Fy) 
    e=(X-np.sum(X*Fx))*(X-np.sum(X*Fx))
    V_glob_X=np.sum(Fx*e)
    e=(Y-np.sum(Y*Fy))*(Y-np.sum(Y*Fy))
    V_glob_Y=np.sum(Fy*e)
    E_cond_X=np.array( [ np.sum(X*(np.array([M[i][j] for i in range(n)]))*(1/Fy[j])) for j in range(m)] )
    E_cond_Y=np.array([ np.sum( Y*((np.array([M[i][j] for j in range(m)])))/Fx[i]) for i in range(n)])
    V_cond_X=np.array( [ np.sum(((X-E_cond_X[j])**2)*(np.array([M[i][j] for i in range(n)]))*(1/Fy[j])) for j in range(m)] )
    V_cond_Y=np.array([ np.sum( ((Y-E_cond_Y[i])**2)*((np.array([M[i][j] for j in range(m)])))/Fx[i]) for i in range(n)])
    e=(E_cond_Y-np.sum(Y*Fy))**2
    v=(Y-np.sum(Y*Fy))**2
    rY_X= ( np.sum(e*Fx)) / (np.sum((v)*Fy))
    e=(E_cond_X-np.sum(X*Fx))**2
    v=(X-np.sum(X*Fx))**2
    rX_Y= ( np.sum(e*Fy) ) / np.sum((v)*Fx)

    text="""
Coefficient de Contingence 	: {}

Coefficient de Pearson		: {} 

Distributions marginales	:
	Pour la caractère X : 	les valeurs de xj      = {}
				les freq de fi.        = {}
	Pour la caractère Y : 	les valeurs de yj      = {}
				les freq de f.j        = {}

Moyennes et Variances Marginales: 
	Pour la caractère X : moy(X)= {} , var(X)= {}
	Pour la caractère Y : moy(Y)= {} , var(Y)= {}

Moy/Var	Conditionnelles 	: 
	Pour la caractère X :
                les valeurs de yj      = {}
                les valeurs de moy(Xj) = {}
                les valeurs de var(Xj) = {}
	Pour la caractère Y :
                les valeurs de xi      = {}
                les valeurs de moy(Yi) = {}
                les valeurs de var(Yi) = {}

Rapports de Corrélation 	:
	la corrélation de Y en X ηY/X = {}
	la corrélation de X en Y ηX/Y = {}

    """.format(Contingence,Pearson ,X,Fx,Y,Fy,E_glob_X ,V_glob_X , E_glob_Y , V_glob_Y ,Y,E_cond_X,V_cond_X ,X,E_cond_Y,V_cond_Y,rY_X , rX_Y)
    print(text)

import numpy as np
from numpy import log as ln
def cov(X,Y,M,Fx,Fy) :
    n=len(M)
    m=len(M[0])
    s=0
    for i in range(n) :
        for j in range(m) :
            s+=M[i][j]*(X[i]-sum(Fx*X))*(Y[j]-sum(Y*Fy))
    return s
def V(X,Fx) :
    return sum(Fx*((X-sum(X*Fx))**2))
def E(X,Fx) :
    return sum(X*Fx)

def REG_Y_f_X_(X,Y,M=[]) :
    X=np.array(X)
    Y=np.array(Y)
    if len(M) == 0 :
        M=np.identity(len(X))
    M=np.array(M)
    M=M/np.sum(M)
    n=len(M)
    m=len(M[0])
    Fx=[ sum(M[i]) for i in range(n)]
    Fy=[ sum([M[i][j] for i in range(n)]) for j in range(m) ]
    
    a=cov(X,Y,M,Fx,Fy)/V(X,Fx)
    b=E(Y,Fy)-a*E(X,Fx)
    r=cov(X,Y,M,Fx,Fy)/np.sqrt(V(X,Fx)*V(Y,Fy))
    regLin_Y_X="Y={}*X+{} avec deg de correlation = {} ".format(a,b,r)

    a=cov(X,ln(Y),M,Fx,Fy)/V(X,Fx)
    k=np.exp(E(ln(Y),Fy)-a*E(X,Fx)) 
    r=cov(X,ln(Y),M,Fx,Fy)/np.sqrt(V(X,Fx)*V(ln(Y),Fy))
    regExp_Y_X="Y={}*exp({}*X) avec deg de correlation = {}".format(k,a,r)

    a=cov(ln(X),ln(Y),M,Fx,Fy)/V(ln(X),Fx) 
    k=np.exp(E(ln(Y),Fy)-a*E(ln(X),Fx)) 
    r=cov(ln(X),ln(Y),M,Fx,Fy)/np.sqrt(V(ln(X),Fx)*V(ln(Y),Fy))
    regEl_Y_X="Y={}*(X^{}) avec deg de correlation = {}".format(k,a,r)

    a=cov(X,ln(Y/(1-Y)),M,Fx,Fy) 
    k=np.exp( -E(ln(Y/(1-Y)),Fy)+a*E(X,Fx))
    r=cov(X,ln(Y*(1/(1-Y))),M,Fx,Fy)/np.sqrt(V(X,Fx)*V(ln(Y/(1-Y)),Fy))
    regLog_Y_X="Y=1/(1+{}*exp(-{}*X)) avec deg de correlation = {}".format(k,a,r)
    text="""
    REGRESSION Y=f(X)       : 
            la droite m.c.o.                :
                 {}

            schémas exponentiels            : 
                {}

            schéma à élasticité constante   : 
                {}
                
            schémas logistique              : 
                {} 
    """.format(regLin_Y_X,regExp_Y_X,regEl_Y_X,regLog_Y_X)
    print(text)

import numpy as np
from numpy import log as ln
def cov(X,Y,M,Fx,Fy) :
    n=len(M)
    m=len(M[0])
    s=0
    for i in range(n) :
        for j in range(m) :
            s+=M[i][j]*(X[i]-sum(Fx*X))*(Y[j]-sum(Y*Fy))
    return s
def V(X,Fx) :
    return sum(Fx*((X-sum(X*Fx))**2))
def E(X,Fx) :
    return sum(X*Fx)
def REG_X_f_Y_(X,Y,M) :
    X=np.array(Y)
    Y=np.array(X)
    if len(M) == 0 :
        M=np.identity(len(X))
    M=np.transpose(np.array(M))
    M=M/np.sum(M)
    n=len(M)
    m=len(M[0])
    Fx=[ sum(M[i]) for i in range(n)]
    Fy=[ sum([M[i][j] for i in range(n)]) for j in range(m) ]
    
    a=cov(X,Y,M,Fx,Fy)/V(X,Fx)
    b=E(Y,Fy)-a*E(X,Fx)
    r=cov(X,Y,M,Fx,Fy)/np.sqrt(V(X,Fx)*V(Y,Fy))
    regLin_Y_X="X={}*Y+{} avec deg de correlation = {} ".format(a,b,r)

    a=cov(X,ln(Y),M,Fx,Fy)/V(X,Fx)
    k=np.exp(E(ln(Y),Fy)-a*E(X,Fx)) 
    r=cov(X,ln(Y),M,Fx,Fy)/np.sqrt(V(X,Fx)*V(ln(Y),Fy))
    regExp_Y_X="X={}*exp({}*Y) avec deg de correlation = {}".format(k,a,r)

    a=cov(ln(X),ln(Y),M,Fx,Fy)/V(ln(X),Fx) 
    k=np.exp(E(ln(Y),Fy)-a*E(ln(X),Fx)) 
    r=cov(ln(X),ln(Y),M,Fx,Fy)/np.sqrt(V(ln(X),Fx)*V(ln(Y),Fy))
    regEl_Y_X="X={}*(Y^{}) avec deg de correlation = {}".format(k,a,r)

    a=cov(X,ln(Y/(1-Y)),M,Fx,Fy) 
    k=np.exp( -E(ln(Y/(1-Y)),Fy)+a*E(X,Fx))
    r=cov(X,ln(Y*(1/(1-Y))),M,Fx,Fy)/np.sqrt(V(X,Fx)*V(ln(Y/(1-Y)),Fy))
    regLog_Y_X="X=1/(1+{}*exp(-{}*Y)) avec deg de correlation = {}".format(k,a,r)
    text="""
    REGRESSION X=f(Y)       : 
            la droite m.c.o.                :
                 {}

            schémas exponentiels            : 
                {}

            schéma à élasticité constante   : 
                {}
                
            schémas logistique              : 
                {} 
    """.format(regLin_Y_X,regExp_Y_X,regEl_Y_X,regLog_Y_X)
    print(text)