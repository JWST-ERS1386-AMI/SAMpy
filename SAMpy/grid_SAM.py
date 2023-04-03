import sys
from emcee.utils import MPIPool
from emcee import PTSampler,EnsembleSampler
import emcee
import os
from numpy import random
import time
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


from ipywidgets import IntProgress
from IPython.display import display as idisplay
import time

def binary_mod(lam,cpuvs,v2uvs,modpars,ang,dokp=False,inc_v2s=True):
    ######make this faster like binary_cps_pix below!!!!
    cpuvs2 = cpuvs/(lam*1.0e-06)/206265*2.0*np.pi
    cpuvs2[:,:,1]*=-1.0

    v2uvs2 = v2uvs/(lam*1.0e-06)/206265*2.0*np.pi
    v2uvs2[:,1]*=-1.0

    npars = len(modpars)
    ncs = int(npars/3)
    pas = np.array([modpars[int(i*3)] for i in range(ncs)])
    ss = np.array([modpars[int(i*3+1)] for i in range(ncs)])
    dms = np.array([modpars[int(i*3+2)] for i in range(ncs)])
    pas_d = np.round(90-ang+pas,3)
    bs = 10**(dms/-2.5)
    for i in range(len(pas_d)):
        while pas_d[i] < -180.0: pas_d[i] += 360
        while pas_d[i] > 180.0: pas_d[i] -= 360
    pas_r = np.radians(pas_d)
    xs = ss*np.cos(pas_r)
    ys = ss*np.sin(pas_r)


    cplist = []
    for tri in cpuvs2:
        tmp = []
        for pix in tri:
            u = pix[0]
            v = pix[1]
            uxvys = u*xs+v*ys
            FT = 1.0 + np.sum(bs*(np.cos(uxvys)+1.0j*np.sin(uxvys)))
            ph = np.angle(FT,deg=1)
            tmp.append(ph)
        cp = np.sum(tmp)
        cplist.append(cp)

    if inc_v2s==True:
        v2list = []
        for bl in v2uvs2:
            u = bl[0]
            v = bl[1]
            uxvys = u*xs+v*ys
            FT = 1.0 + np.sum(bs*(np.cos(uxvys)+1.0j*np.sin(uxvys)))
            amp = np.abs(FT)/(1.0+np.sum(bs))
            v2list.append(amp**2)
        #if dokp==True:
        #    CP = np.matrix(np.array(mlist).reshape([len(mlist),1]))
        #    kps = np.array(kproj*CP).flatten()
        #    return kps
        #else: return mlist
        return np.array(cplist),np.array(v2list)
    else: 
        return np.array(cplist)

    
def binary_cps_pix(lam,cpuvs2,modpars,ang,dokp=False):
    
    #cpuvs2 = cpuvs/(lam*1.0e-06)/206265*2.0*np.pi
    #cpuvs2[:,:,1]*=-1.0

    npars = len(modpars)
    ncs = int(npars/3)
    pas = np.array([modpars[int(i*3)] for i in range(ncs)])
    ss = np.array([modpars[int(i*3+1)] for i in range(ncs)])
    dms = np.array([modpars[int(i*3+2)] for i in range(ncs)])
    pas_d = np.round(90-ang+pas,3)
    bs = 10**(dms/-2.5)
    for i in range(len(pas_d)):
        while pas_d[i] < -180.0: pas_d[i] += 360
        while pas_d[i] > 180.0: pas_d[i] -= 360
    pas_r = np.radians(pas_d)
    xs = ss*np.cos(pas_r)
    ys = ss*np.sin(pas_r)
    
    cplist = []
    for tri in cpuvs2:
        us = tri[:,:,0]
        vs = tri[:,:,1]
        uxvys = np.array([us*xs[ii]+vs*ys[ii] for ii in range(len(xs))])
        FT = np.sum(1.0+np.array([bs[ii]*(np.cos(uxvys[ii])+1.0j*np.sin(uxvys[ii])) for ii in range(len(bs))]),axis=0)
        cp = np.angle(np.prod(FT,axis=-1),deg=1)
        cplist.append(cp)       
    return np.array(cplist)
    
    
    

def binary_chi2(cpdata,cperrs,cpuvs,v2data,v2errs,v2uvs,rotlist,pars,lam,inc_v2s=True):
    chi = []
    cpalls = []
    v2alls = []
    for a in range(len(rotlist)):
        if inc_v2s==True:
            cps,v2s = binary_mod(lam,cpuvs,v2uvs,pars,rotlist[a],dokp=False,inc_v2s=inc_v2s)
            cpalls.append(cps)
            v2alls.append(v2s)
        else:
            cps = binary_mod(lam,cpuvs,v2uvs,pars,rotlist[a],dokp=False,inc_v2s=inc_v2s)
            cpalls.append(cps)
    if inc_v2s==True:
        chi = np.sum([
                np.sum((np.array(cpdata)-np.array(cpalls))**2/np.array(cperrs)**2),
                np.sum((np.array(v2data)-np.array(v2alls))**2/np.array(v2errs)**2)
                ])
    else:
        chi = np.sum([
                np.sum((np.array(cpdata)-np.array(cpalls))**2/np.array(cperrs)**2)])
    return chi




def gen_chi2_grid(cpdata,cperrs,cpuvs,v2data,v2errs,v2uvs,rotlist,lam,
                  verbose=False,inc_v2s=True,
                  npas=11,nseps=51,ndms=51,
                 maxsep=0.5,maxcont=10.0):
    pas = np.linspace(0.0,360.0,npas)
    seps = np.linspace(0.0,maxsep,nseps)
    dms = np.linspace(0.0,maxcont,ndms)

    g2 = [[[[] for sep in seps] for dm in dms] for pa in pas]
    coords2 = [[[[] for sep in seps] for dm in dms] for pa in pas]
    f = IntProgress(min=0, max=len(dms)*len(seps)*len(pas)) # instantiate the bar
    display(f) # display the bar
    f.value=0
    for i in range(len(dms)):
        dm = dms[i]
        if verbose==True: print(dm)
        for j in range(len(seps)):
            sep = seps[j]
            ctmp = []
            for k in range(len(pas)):
                pa = pas[k]
                c2 = binary_chi2(cpdata,cperrs,cpuvs,v2data,v2errs,v2uvs,rotlist,[pa,sep,dm],lam,inc_v2s=inc_v2s)
                ctmp.append(c2)
                g2[k][i][j]=c2
                coords2[k][i][j] = [pas[k],dms[i],seps[j]]
                f.value+=1
    return np.array(g2),np.array(coords2)


def make_cc(g,coords,cpdata,v2data,inc_v2s=True,filename=None):
    if inc_v2s==True:
        dof = len(cpdata.flatten())+len(v2data.flatten())-3.0
    else:
        dof = len(cpdata.flatten())-3.0
    g2 = np.mean(g,axis=0)
    g2 = g2/np.min(g2)*dof
    g2 = g2-g2[0,0]
    seps = np.unique(coords[:,:,:,-1])
    dms = np.unique(coords[:,:,:,-2])
    print(seps)
    print(dms)
    X,Y= np.meshgrid(seps,dms)
    f = plt.figure(figsize=(10,7))
    plt.subplots_adjust(bottom=0.15)
    #plt.title('AB Dor Contrast')
    #ax = f.add_subplot(111)
    CS=plt.contour(X,Y,g2,levels=[np.min(g2),1.0,4.0,9.0,16.0,25.0],colors='k')
    CF=plt.contourf(X,Y,g2,levels=[np.min(g2),1.0,4.0,9.0,16.0,25.0],alpha = 0.8)
    f.patch.set_facecolor('white')
    dat0= CS.allsegs
    #plt.xlim(0.01,0.25)
    #slist = [3,5]
    #for s in [5]:
    #    dats = dat0[s-1][0]
    #    print dats.shape
    #    np.savetxt(odir+'cc_s'+str(s)+'.txt',np.array(dats))
    #cc3o = np.loadtxt(odir+'cc_s3.txt')
    #cc5o = np.loadtxt(odir+'cc_s5.txt')
    #plt.plot(cc3o[:,0],cc3o[:,1],'k--')
    #plt.plot(dats[:,0],dats[:,1],'k')
    plt.ylim(11,2)
    plt.grid(which='major')
    #plt.title('AB Dor Cal Contrast Curve')
    plt.xlabel('Separation (arcsec)',fontsize=14)
    plt.ylabel('Contrast (mag)',fontsize=14)
    cbar=plt.colorbar(CF)
    cbar.ax.yaxis.set_ticklabels(['',r'$1 \sigma$',r'$2 \sigma$',r'$3 \sigma$',r'$4 \sigma$',r'$5 \sigma$'],fontsize=12)
    if filename!=None:
        plt.savefig(filename)
    #plt.savefig('example_hip65426_cc.pdf',dpi=300)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return f,dat0

def proc_binary_grid(grid,coords,cps,v2s,filename=None,rescale=True,inc_v2s=True):
    bfit = coords[np.where(grid==np.min(grid))][0]
    bmod = [bfit[0],bfit[2],bfit[1]]
    dof = len(cps.flatten())+len(v2s.flatten()) - 3.0
    if inc_v2s==False: 
        dof = len(cps.flatten())
    bdm = np.where(grid==np.min(grid))[1][0]
    if rescale==True: grid_r = grid*dof/np.min(grid)
    else: grid_r = grid
    gridslice = grid_r[:,bdm,:]
    coordsslice = coords[:,bdm,:]
    PAs_g = coordsslice[:,:,0]
    seps_g = coordsslice[:,:,2]
    xs = -np.sin(np.radians(PAs_g))*seps_g
    ys = np.cos(np.radians(PAs_g))*seps_g
    points = np.array([[xs[i,j],ys[i,j]] for i in range(len(xs)) for j in range(len(xs[i]))])
    vals = np.array([gridslice[i,j] for i in range(len(xs)) for j in range(len(xs[i]))])
    xs = np.linspace(-np.max(seps_g),np.max(seps_g),100)
    X,Y = np.meshgrid(xs,xs)
    gint = griddata(points,vals,(X,Y))
    f = plt.figure(figsize=(6,5))
    f.add_subplot(111)
    pcm=plt.pcolormesh(X,Y,gint/np.nanmin(gint)*32,rasterized=True,cmap=plt.get_cmap('cubehelix'))
    plt.colorbar(pcm)
    plt.scatter(-np.sin(np.radians(bmod[0]))*bmod[1],np.cos(np.radians(bmod[0]))*bmod[1],marker='x',color='w',s=100)
    #plt.contour(X,Y,gint/np.min(gint)*32,levels=[gint/np.min(gint)*32+3.53],colors='w')
    plt.xticks([-0.4,-0.2,0.0,0.2,0.4],['0.4','0.2','0.0','-0.2','-0.4'])
    plt.title('r$\chi^2$ Surface: Contrast = '+str(bmod[-1])+' mag')
    plt.xlabel(r'$\Delta$RA (arcsec)')
    plt.ylabel(r'$\Delta$DEC (arcsec)')
    if filename!=None:
        plt.savefig(filename)
    plt.show()
    return bmod,np.min(grid_r)
    
