import numpy as np
import matplotlib.pyplot as plt


def func(B,Bvar,time):
    ret=0
    var=0
    for x in range(len(B)):
        coeff = B[x,0]
        ret+=coeff*time**x
        var+=time**x*Bvar[x]
    return ret,var

def poly_ps(tcps,ccps,tt,ctin,cords,datatype,tvar=[],cvar=[],display=False):
    calpars = []
    cp_cal = []
    v_cal = []
    v_cal_scat = []
    chi = 0
    c = 0
    print('cords = '+str(cords))
    #tcps should have shape [npointings,ncps]
    #ccps should have shape [ncalpointings,ncps]
    for tind in range(len(tcps[0])):
        X = []
        data = np.array([ccps[x,tind] for x in range(len(ccps))])
        if len(cvar) > 0:
            var = np.array([cvar[x,tind] for x in range(len(ccps))])
            W = np.matrix(np.diag(1/np.array(var)))
        else: W = np.diag(np.ones(len(ccps)))
        
        ct = np.array([ctin[x] for x in range(len(ctin))])
        X = [[i**order for order in range(cords+1)] for i in ct]
        X = np.matrix(X)
        Y = np.matrix(data).reshape(len(data),1)
        B = (X.T*W*X).I*X.T*W*Y
        Bvar = np.diag((X.T*W*X).I)
        calpars.append([np.array(B)[:,0],Bvar])
        
        at = [t for t in ct]
        for t in tt: at.append(t)
        modts = np.linspace(np.min(at),np.max(at),100)
        model = [func(B,Bvar,i)[0] for i in modts]
        fake_cals = [func(B,Bvar,i)[0] for i in ct]

        chi+=np.sum((fake_cals-data)**2)
        ts = [func(B,Bvar,i)[0] for i in tt]
        tsv = [func(B,Bvar,i)[1] for i in tt]
        avgs = [np.mean(ccps[:,tind]) for x in modts]


        if display==True:
            tmin = np.min([np.min(ct),np.min(tt)])
            f = plt.figure(figsize=(8,5))
            plt.title(' CP Index '+str(tind)+'; Polycal Order '+str(cords),fontsize=16)
            plt.plot((modts-tmin)/3600,model,color='k',label='Model',zorder=-1)
            plt.plot([(tt-tmin)/3600,(tt-tmin)/3600],[ts,tcps[:,tind]],'k--',lw=0.5)
            plt.scatter((ct-tmin)/3600,data,facecolors='grey',edgecolors='k',label='Ref. PSF',s=100)
            plt.scatter((tt-tmin)/3600,tcps[:,tind],facecolors='purple',edgecolors='k',label='Science',s=100)
            plt.legend(fontsize=14)
            plt.ylabel(r'CP ($^\circ$)',fontsize=14)
            plt.xlabel('Time (hours)',fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.show()

        if datatype=='cps':
            cp_cal.append(tcps[:,tind]-ts)
            if len(tvar)>0: v_cal.append(tvar[:,tind]+tsv)
            else: v_cal.append(tsv)
        if datatype=='v2s':
            cp_cal.append(tcps[:,tind]/ts)
            if len(tvar)>0: v_cal.append(tvar[:,tind]+tsv)
            else: v_cal.append(tsv)
        v_cal_scat.append([np.std(ccps[:,tind]) for x in tcps[:,tind]])
    cp_cal = np.array(cp_cal)
    v_cal = np.array(v_cal)
    v_cal_scat = np.array(v_cal_scat)
    calpars = np.array(calpars)
    return cp_cal,v_cal,v_cal_scat,calpars