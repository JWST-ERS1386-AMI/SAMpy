import sys
from emcee.utils import MPIPool
from emcee import PTSampler,EnsembleSampler
import emcee
import os
from numpy import random
import time
import astropy.io.fits as pyfits
import numpy as np

from ipywidgets import IntProgress
from IPython.display import display
import time
import matplotlib.pyplot as plt
import corner


def binary_phase(lam,uvs,v2uvs,modpars,ang,inc_v2s=True):
    uvs2 = uvs/(lam*1.0e-06)/206265*2.0*np.pi
    uvs2[:,:,1]*=-1.0
    
    v2uvs2 = v2uvs/(lam*1.0e-06)/206265*2.0*np.pi
    v2uvs2[:,1]*=-1.0

    npars = len(modpars)
    ncs = int(npars/3)
    pas = np.array([modpars[i*3] for i in range(ncs)])
    ss = np.array([modpars[i*3+1] for i in range(ncs)])
    dms = np.array([modpars[i*3+2] for i in range(ncs)])
    pas_d = np.round(90-ang+pas,3)
    bs = 10**(dms/-2.5)
    for i in range(len(pas_d)):
        while pas_d[i] < -180.0: pas_d[i] += 360
        while pas_d[i] > 180.0: pas_d[i] -= 360
    pas_r = np.radians(pas_d)
    xs = ss*np.cos(pas_r)
    ys = ss*np.sin(pas_r)
    mlist = []

    for tri in uvs2:
        #tmp = []
        cp=0.0
        for pix in tri:
            u = pix[0]
            v = pix[1]
            uxvys = u*xs+v*ys
            FT = 1.0 + np.sum(bs*(np.cos(uxvys)+1.0j*np.sin(uxvys)))
            ph = np.angle(FT,deg=1)
            #tmp.append(ph)
            cp+=ph
        #cp = np.sum(tmp)
        mlist.append(cp)
    if inc_v2s==True:
        blist = []
        for bl in v2uvs2:
            u = bl[0]
            v = bl[1]
            uxvys = u*xs+v*ys
            FT = 1.0 + np.sum(bs*(np.cos(uxvys)+1.0j*np.sin(uxvys)))
            v2 = (np.abs(FT)/(1.0+np.sum(bs)))**2
            blist.append(v2)
        return np.array(mlist),np.array(blist)
    else: return np.array(mlist)


def lnlike(pars,v2d,v2de,bl_uvs_2,cpd,cpde,cp_uvs_2,angs,inc_v2s,lam):
    cpalls,v2alls = [],[]
    for ang in angs:
        if inc_v2s==True:
            cptmp,v2tmp = binary_phase(lam,cp_uvs_2,bl_uvs_2,pars,ang,inc_v2s=inc_v2s)
            v2alls.append(v2tmp)
        else: 
            cptmp = binary_phase(lam,cp_uvs_2,bl_uvs_2,pars,ang,inc_v2s=inc_v2s)
        cpalls.append(cptmp)
    chic = np.sum((np.array(cpalls)-np.array(cpd))**2/np.array(cpde)**2)
    if inc_v2s==True: chiv = np.sum((np.array(v2alls)-np.array(v2d))**2/np.array(v2de)**2)
    else: chiv=0
    return -0.5*(chic+chiv)



def lnprior(modpars):
    ncs = int(len(modpars)/3)
    pas = np.array([modpars[i*3] for i in range(ncs)])
    ss = np.array([modpars[i*3+1] for i in range(ncs)])
    dms = np.array([modpars[i*3+2] for i in range(ncs)])
    for pa in pas:
        if -180.0 > pa: return -np.inf
        if 180.0 <= pa: return -np.inf
    for s in ss:
        if s < 0.0: return -np.inf
        if s > 0.4: return -np.inf
    for dm in dms:
        if dm < 0.0: return -np.inf
        if dm > 10.0: return -np.inf
    for i in range(len(pas)):
        for j in range(i):
            if j > i:
                if pas[i] > pas[j]: return -np.inf
    return 0.0

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    #print(lp+lnlike)
    return lp + lnlike(theta)



def run_PT_emcee(v2data,v2errs,v2uvs,cpdata,cperrs,cpuvs,angs,lam,
                 inc_v2s=False,ndim=3,nwalkers=100,ntemps=10,NTHREADS=1,writeint=100,
                 odir='./',NITS=1000,verbose=False,overwrite=True,suff=''):
    start = [-180.0,0,0]
    scales = [350.0,0.4,10.0]
    p0 = [[np.array(start)+np.array(scales)*np.random.uniform(low=0.0,high=1.0,size=len(scales)) for i in range(nwalkers)] for j in range(ntemps)]

    p0 = np.array(p0)
    for j in range(ntemps):
        for i in range(nwalkers):
            modpars = p0[j,i]
            ncs = int(len(modpars)/3)
            pas = np.array([modpars[int(k*3)] for k in range(ncs)])
            pasort = np.sort(pas)
            painds = [int(k*3) for k in range(ncs)]
            for k in range(len(painds)):
                modpars[painds[k]] = pasort[k]
            p0[j,i] = modpars


    sampler = PTSampler(ntemps,nwalkers,ndim,lnlike,lnprior,loglargs=[v2data,v2errs,v2uvs,cpdata,cperrs,cpuvs,angs,inc_v2s,lam],threads=NTHREADS)


    if overwrite==False:
        if os.path.isfile(odir+'lnls'+suff+'.fits')==True:
            plist = list(pyfits.getdata(odir+'chain'+suff+'.fits'))
            lnplist = list(pyfits.getdata(odir+'lnps'+suff+'.fits'))
            lnllist = list(pyfits.getdata(odir+'lnls'+suff+'.fits'))
            count = len(plist)
            p0 = plist[-1]
        else:
            plist = []
            lnplist = []
            lnllist = []
            count = 0
    else:
        plist = []
        lnplist = []
        lnllist = []
        count = 0
    count2 = 0
    
    print('about to sample')
    print('runnning steps '+str(count)+' to '+str(NITS))
    f = IntProgress(min=count, max=NITS) # instantiate the bar
    display(f) # display the bar
    f.value=count
    
    for res in sampler.sample(p0, iterations=NITS-count, storechain=False):
        if verbose==True:
            if count%10==True: 
                print(count)
        p,lnp,lnl = res
        plist.append(p)
        lnplist.append(lnp)
        lnllist.append(lnl)
        f.value += 1
        count+=1
        count2+=1
        if count2==writeint:
            pyfits.writeto(odir+'chain'+suff+'.fits',np.array(plist),overwrite=True)
            pyfits.writeto(odir+'lnps'+suff+'.fits',np.array(lnplist),overwrite=True)
            pyfits.writeto(odir+'lnls'+suff+'.fits',np.array(lnllist),overwrite=True)
            count2 = 0
    pyfits.writeto(odir+'chain'+suff+'.fits',np.array(plist),overwrite=True)
    pyfits.writeto(odir+'lnps'+suff+'.fits',np.array(lnplist),overwrite=True)
    pyfits.writeto(odir+'lnls'+suff+'.fits',np.array(lnllist),overwrite=True)
    return plist,lnplist,lnllist




def plot_corner(chain,lnls,ntemps=10,nwalkers=100,burnin=100,title='',truths=[],
                fname='',pnames='',smooth=False,figsize=(7,6)):
    lnls2 = lnls.reshape([ntemps,nwalkers,len(lnls)])
    dim = len(chain[0,0,0])
    logl = 0
    logp = 0
    sampler = emcee.PTSampler(ntemps, nwalkers, dim, logl, logp)
    logz,dlogz = sampler.thermodynamic_integration_log_evidence(lnls2,fburnin=0.0)
    print(logz,dlogz)
    print(chain.shape)

    chain2 = chain[burnin:,0,:].reshape([(len(chain)-burnin)*100,dim])
    print(chain2.shape)
    if pnames == '':
        pnames = [r'PA ($^\circ$)',r'$\rho$ (arcsec)',r'$\Delta$ (mag)']
    #pnames = ['PA1','s1','dm1','PA2','s2','dm2','PA3','s3','dm3']
    fig = plt.figure(figsize=figsize)
    if len(truths)>0:
        corner.corner(chain2,labels=pnames,fig=fig, max_n_ticks=4,truths=truths,fontsize=12,smooth=smooth)
    else:
        corner.corner(chain2,labels=pnames,fig=fig, max_n_ticks=4,fontsize=12,smooth=smooth)
    plt.subplots_adjust(top=0.9)
    fig.suptitle(title,fontsize=18)
    if fname!='': plt.savefig(fname,dpi=300)
    plt.show()
    mpars = chain[np.where(lnls==np.max(lnls))][0]
    #print(mpars)
    #chain2[:,0]-=mpars[0]
    print(np.max(lnls)*-2)
    print(mpars)

    confs = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                         zip(*np.percentile(chain2, [16, 50, 84],
                                            axis=0))))
    cpars = []
    for i in range(len(confs)):
        print(confs[i])
        cpars.append(confs[i][0])
    #print '--------------------------------------------------------------------'
    return mpars,cpars


