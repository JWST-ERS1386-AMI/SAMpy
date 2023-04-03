from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import map_coordinates
from .red_SAM import supergauss_fracw
import cmath
import math
import os
import numpy as np
import astropy.io.fits as pyfits
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy import interpolate
import pdb
import copy
import h5py
from .utils import *






def gen_window(w,boxsize,lam,ps):
    """
    generates window functions:
        -'sg' = supergauss
        -'h' = hanning
        -'nw' = no window

    lam is wavelength in um
    ps is platescale in arcsec/pix
    """
    if w=='sg':
        wwidth = lam*1.0e-6*206265./ps*0.65
        hbox = supergauss_fracw(wwidth,0.95,8.0,boxsize)
    if w=='h':
        hbox=np.outer(np.hanning(boxsize),np.hanning(boxsize))
    if w=='nw':
        hbox = np.ones([boxsize,boxsize])
    return hbox

def fft_image(im,nx,ny):
    """
    pads a single image to 1024 by 1024 and subframes it
    """
    boxsize = len(im)
    imb = np.pad(im,(((ny-len(im))//2,(ny-len(im))//2),
                     ((nx-len(im[0]))//2,(nx-len(im[0]))//2)))
    fft=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(imb)))
    return fft

def make_pspec(im,nx,ny):
    """
    generates a power spectrum for a single image
    """
    fft = fft_image(im,nx,ny)
    ps2 = abs(fft)**2
    return ps2

def make_cpcoords(mdir):
    """
    loads cp splodge locations in pixels (looks for either file name)
    """
    try: ccs = np.array(pyfits.getdata(mdir+'cp_pix.fits'))
    except IOError: ccs = np.array(pyfits.getdata(mdir+'cp_coords.fits'))
    return ccs

def make_v2coords(mdir):
    """
    loads v2 splodge locations in pixels (looks for either file name)
    """
    try: bcs = np.array(pyfits.getdata(mdir+'bl_pix.fits'))
    except IOError: bcs = np.array(pyfits.getdata(mdir+'bl_coords.fits'))
    return bcs



def make_cviscoords(mdir):
    """
    loads v2 splodge locations in pixels (looks for either file name)
    """
    try: bcs = np.array(pyfits.getdata(mdir+'cvis_pix.fits'))
    except IOError: bcs = np.array(pyfits.getdata(mdir+'cvis_coords.fits'))
    return bcs


def find_v2_pix(mdir,v2type='multi'):
    """
    loads sampling coordinates for either single pixel
    or "Monnier" (using many pixels) mode for v2s
    """
    bcs = make_v2coords(mdir)
    bclist = []
    if v2type=='single':
        for i in range(len(bcs)):
            bclist.append(np.array([np.round(bcs[i])],dtype='int'))
    if v2type=='multi':
        for i in range(len(bcs)):
            ps = pyfits.getdata(mdir+'v2_ind'+str(i)+'.fits')
            bclist.append(ps)
    return bclist


def find_cvis_pix(mdir,cvistype='multi'):
    """
    loads sampling coordinates for either single pixel
    or "Monnier" (using many pixels) mode for v2s
    """
    bcs = make_cviscoords(mdir)
    bclist = []
    if cvistype=='single':
        for i in range(len(bcs)):
            bclist.append(np.array([np.round(bcs[i])],dtype='int'))
    if cvistype=='multi':
        for i in range(len(bcs)):
            ps = pyfits.getdata(mdir+'cvis_ind'+str(i)+'.fits')
            bclist.append(ps)
    return bclist

def find_tris_multi_2(ccs,cdir,ny=256,nx=256,meters=False,uv=False,pscam=0.065,lamc=3.8,redo_calc=False): #Function 2 definition
    """
    calculates sampling coordinates for "Monnier"
    closure phase calculation method

    -the first time this runs it finds all triangles of pixels
    that connect the three splodges for a single closing triangle
    and that also satisfy u1,v1+u2,v2+u3,v3 = 0,0 and saves the
    output to the mask directory
    -as-is, if you want to adjust your mask, you need to manually
    delete the "cpsamp" files from the mask directory, sorry.
    """
    ccall = []
    freqs = np.fft.fftshift(np.fft.fftfreq(nx))
    for x in range(len(ccs)):
        c = ccs[x]
        ifile = cdir+'cpsamp_ind_'+str(x)+'.fits'
        if (os.path.isfile(ifile)==False) or (redo_calc==True): #Executed during first run
            print ('\n Calculating triangle: '+str(x+1)+' of '+str(len(ccs)))
            ib = cdir+'ind'+str(x)+'_vert'
            p0 = pyfits.getdata(ib+'0.fits')
            p1 = pyfits.getdata(ib+'1.fits')
            p2 = pyfits.getdata(ib+'2.fits')
            g = []
            for pp1 in tqdm(p0):
                if pp1[1] not in [8,16,24,40,56,72]:
                    for pp2 in p1:
                        if pp2[1] not in [8,16,24,40,56,72]:
                    #cy=ny//2
                    #cx=nx//2
                            for pp3 in p2:
                                if pp3[1] not in [8,16,24,40,56,72]:
                                    sumx = freqs[pp1[0]]+freqs[pp2[0]]+freqs[pp3[0]]
                                    sumy = freqs[pp1[1]]+freqs[pp2[1]]+freqs[pp3[1]]
                                    if sumx==0:
                                        if sumy==0:
                                            g.append(np.array([pp1,pp2,pp3],dtype='int'))
            print(len(g))
            print('---------------')
            g = np.array(g)
            pyfits.writeto(ifile,g,overwrite=True)
        else: g = pyfits.getdata(ifile)
        ccall.append(g)
    if meters: 
        ###returns u,v that go with pixel sampling measured in m
        ###need to flip us to get to true u,v coords
        ccall2 = np.array(copy.deepcopy(ccall))
        psc = 1.0/(float(nx)*pscam)*206265.0*lamc*1e-06
        ccall2 = ccall2 - int(ny/2)
        ccall2 = ccall2*psc
        if uv: ###flipping u to get to real u,v
            for i in range(len(ccall2)):
                ccall2[i][:,:,0] = -ccall2[i][:,:,0]
        return ccall2
    return np.array(ccall)


def find_tris_multi(ccs,cdir,ny=256,nx=256,meters=False,uv=False,pscam=0.065,lamc=3.8,redo_calc=False): #Function 2 definition
    """
    calculates sampling coordinates for "Monnier"
    closure phase calculation method

    -the first time this runs it finds all triangles of pixels
    that connect the three splodges for a single closing triangle
    and that also satisfy u1,v1+u2,v2+u3,v3 = 0,0 and saves the
    output to the mask directory
    -as-is, if you want to adjust your mask, you need to manually
    delete the "cpsamp" files from the mask directory, sorry.
    """
    ccall = []
    for x in range(len(ccs)):
        c = ccs[x]
        ifile = cdir+'cpsamp_ind_'+str(x)+'.fits'
        if (os.path.isfile(ifile)==False) or (redo_calc==True): #Executed during first run
            print ('\n Calculating triangle: '+str(x+1)+' of '+str(len(ccs)))
            ib = cdir+'ind'+str(x)+'_vert'
            p0 = pyfits.getdata(ib+'0.fits')
            p1 = pyfits.getdata(ib+'1.fits')
            p2 = pyfits.getdata(ib+'2.fits')
            g = []
            for pp1 in tqdm(p0):
                for pp2 in p1:
                    cy=ny//2
                    cx=nx//2
                    dpp1 = pp1 - np.array([cy,cx])
                    dpp2 = pp2 - np.array([cy,cx])
                    dpp3 = -dpp1-dpp2
                    pp3 = dpp3 + np.array([cy,cx])      
                    t1 = np.where(p2[:,0]==pp3[0])
                    t2 = np.where(p2[:,1]==pp3[1])
                    inlist = False
                    for t in t1[0]:
                        if t in t2[0]: inlist=True
                    if inlist==True:
                        g.append(np.array([pp1,pp2,pp3],dtype='int'))
            g = np.array(g)
            pyfits.writeto(ifile,g,overwrite=True)
        else: g = pyfits.getdata(ifile)
        ccall.append(g)
    if meters: 
        ###returns u,v that go with pixel sampling measured in m
        ###need to flip us to get to true u,v coords
        ccall2 = np.array(copy.deepcopy(ccall))
        psc = 1.0/(float(nx)*pscam)*206265.0*lamc*1e-06
        ccall2 = ccall2 - int(ny/2)
        ccall2 = ccall2*psc
        if uv: ###flipping u to get to real u,v
            for i in range(len(ccall2)):
                ccall2[i][:,:,0] = -ccall2[i][:,:,0]
        return ccall2
    return np.array(ccall)

def make_blens(mdir):
    """
    calculates baseline lengths from uv sampling coordinates
    for plotting purposes
    """
    buvs = np.array(pyfits.getdata(mdir+'bl_uvs.fits'))
    blens = np.sqrt(buvs[:,0]**2+buvs[:,1]**2)
    return blens


def make_cplens(mdir):
    """
    calculates baseline lengths from uv sampling coordinates
    for plotting purposes
    """
    cpuvs = np.array(pyfits.getdata(mdir+'cp_uvs.fits'))
    cplens = np.sqrt(cpuvs[:,:,0]**2+cpuvs[:,:,1]**2)
    return cplens

def calc_cps_single_DFT(ims,mdir,nx=256,ny=256,display=False,useW=True):
    """
    calculate closure phases using single pixels at the center of each splodge
    """
    
    cpDFTmat_Re,cpDFTmat_Im = pyfits.getdata(mdir+'cpDFTmat_sing_'+str(nx)+'.fits')
    IM2CT = cpDFTmat_Re + cpDFTmat_Im*1.0j
    bs_all = []
    imcount=0
    for i in tqdm(ims):
        ctphis = np.dot(IM2CT,i.flatten()) / np.sum(i.flatten())
        bispecs = []
        for j in range(len(ctphis)//3):
            bispec = np.prod([ctphis[j*3+x] for x in range(3)])
            bispecs.append(bispec)
        bs_all.append(bispecs)
        imcount+=1
    cas = np.angle(bs_all,deg=1)
    Aas = np.abs(bs_all)
    cps = np.angle(np.nanmean(bs_all,axis=0),deg=1)
    cov,var,stdE = gen_cov(cps,cas,W=Aas,useW=useW)
    return bs_all, cps, cov, var, stdE

def calc_cps_single(ims,mdir,nx=256,ny=256,display=False,useW=True):
    """
    calculate closure phases using single pixels at the center of each splodge
    """
    ccs = np.array(np.round(make_cpcoords(mdir),1),dtype='int')
    bs_all = []
    imcount=0
    for i in tqdm(ims):
        FT = fft_image(i,nx,ny)
        tmp = np.array([np.prod([FT[tri[j,1],tri[j,0]] for j in range(3)]) for tri in ccs])
        bs_all.append(tmp)
        if display==True:
            if imcount==0:
                plt.imshow(np.abs(FT)**0.1)
                plt.scatter(ccs[:,:,0],ccs[:,:,1],edgecolors='k',facecolors='None')
                rinds=np.random.choice(len(ccs),5)
                for ii in rinds:
                    plt.plot([ccs[ii,jj,0] for jj in [0,1,2,0]],[ccs[ii,jj,1] for jj in [0,1,2,0]],c='w')
                plt.show()
        imcount+=1
    cas = np.angle(bs_all,deg=1)
    Aas = np.abs(bs_all)
    cps = np.angle(np.nanmean(bs_all,axis=0),deg=1)
    cov,var,stdE = gen_cov(cps,cas,W=Aas,useW=useW)
    return bs_all, cps, cov, var, stdE


def calc_cps_multi_image(image,gcs,imcount,nx=256,ny=256,display=False,save=False,fout=None):
    FT = fft_image(image,nx,ny)
    bs = []
    for ind in range(len(gcs)):
        gc = gcs[ind]
        if display==True:
            f=plt.figure(figsize=(5,5))
            ax = f.add_subplot(111)
            plt.imshow(np.abs(FT)**0.1)
            tp = np.mean(gc,axis=0)
            plt.scatter(gc[:,:,0],gc[:,:,1],edgecolors='k',facecolors='None')
            rinds=np.random.choice(len(gc),5)
            for ii in rinds:
                plt.plot([gc[ii,jj,0] for jj in [0,1,2,0]],[gc[ii,jj,1] for jj in [0,1,2,0]],c='w')
            ax.set_yticks([])
            ax.set_xticks([])
            #plt.ylim(20,237)
            #plt.xlim(20,237)
            #plt.savefig('/Users/stephsallum/Dropbox/Talks/220719_SPIE/cp_sampling.pdf')
            plt.show()
                #stop
        tomean = []
        for p in range(len(gc)):
            for j in range(3):
                if j==0:
                    tomult = FT[gc[p,j,1],gc[p,j,0]]/FT[ny//2,nx//2]
                else: tomult*=FT[gc[p,j,1],gc[p,j,0]]/FT[ny//2,nx//2]
            tomean.append(tomult)
        bis_tmp = np.mean(tomean)
        if save: 
            fout['int'+str(imcount)+'/tri'+str(ind)] = tomean ###all pixel triangles for one triangle
        bs.append(bis_tmp)
    return np.array(bs)

def calc_cps_multi_groupimage(image,gcs,imcount,groupcount,nx=256,ny=256,display=False,save=False,fout=None):
    FT = fft_image(image,nx,ny)
    bs = []
    for ind in range(len(gcs)):
        gc = gcs[ind]
        if display==True:
            f=plt.figure(figsize=(5,5))
            ax = f.add_subplot(111)
            plt.imshow(np.abs(FT)**0.1)
            tp = np.mean(gc,axis=0)
            plt.scatter(gc[:,:,0],gc[:,:,1],edgecolors='k',facecolors='None')
            rinds=np.random.choice(len(gc),5)
            for ii in rinds:
                plt.plot([gc[ii,jj,0] for jj in [0,1,2,0]],[gc[ii,jj,1] for jj in [0,1,2,0]],c='w')
            ax.set_yticks([])
            ax.set_xticks([])
            #plt.ylim(20,237)
            #plt.xlim(20,237)
            #plt.savefig('/Users/stephsallum/Dropbox/Talks/220719_SPIE/cp_sampling.pdf')
            plt.show()
                #stop
        tomean = []
        for p in range(len(gc)):
            for j in range(3):
                if j==0:
                    tomult = FT[gc[p,j,1],gc[p,j,0]]/FT[ny//2,nx//2]
                else: tomult*=FT[gc[p,j,1],gc[p,j,0]]/FT[ny//2,nx//2]
            tomean.append(tomult)
        bis_tmp = np.mean(tomean)
        if save: 
            fout['int'+str(imcount)+'/group'+str(groupcount)+'/tri'+str(ind)] = tomean ###all pixel triangles for one triangle
        bs.append(bis_tmp)
    return np.array(bs)

def calc_cps_multi(ims,cdir,display=True,nx=256,ny=256,useW=True,save_allpix=False,filename='',redo_calc=False): #Master function
    """
    calculates closure phases using multiple pixel triangles
    for each triangle of baselines
    """
    ccs = make_cpcoords(cdir)
    gcs = find_tris_multi(ccs,cdir,redo_calc=redo_calc,nx=nx,ny=ny)
    #gcs = find_tris_multi_2(ccs,cdir,redo_calc=redo_calc,nx=nx,ny=ny)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    else:
        fout = None
    bs_all = []
    for xx,i in tqdm(enumerate(ims)):        
        bs = calc_cps_multi_image(i,gcs,xx,nx=nx,ny=ny,display=display,save=save_allpix,fout=fout)
        display=False
        bs_all.append(bs)
    cas = np.angle(bs_all,deg=1)
    Aas = np.abs(bs_all)
    Tas = np.abs(np.mean(bs_all,axis=0))
    cps = np.angle(np.mean(bs_all,axis=0),deg=1)
    if save_allpix: fout.close()
    if len(ims)>1: cov,var,stdE = gen_cov(cps,cas,W=Aas,useW=useW) 
    else: cov,var,stdE = None,None,None
    return np.array([bs_all,cps,Tas,cov,var,stdE])


def calc_cps_multi_groups(ims,cdir,display=True,nx=256,ny=256,useW=True,save_allpix=False,filename='',redo_calc=False): #Master function
    """
    calculates closure phases using multiple pixel triangles
    for each triangle of baselines
    """
    ccs = make_cpcoords(cdir)
    gcs = find_tris_multi(ccs,cdir,redo_calc=redo_calc,nx=nx,ny=ny)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    else:
        fout = None
    bs_all = []
    imcount=0
    for cube in tqdm(ims):  
        groupcount=0
        tmp = []
        for im in cube:
            bs = calc_cps_multi_groupimage(im,gcs,imcount,groupcount,
                                           nx=nx,ny=ny,display=display,
                                           save=save_allpix,fout=fout)
            display=False
            tmp.append(bs)
            groupcount+=1
        imcount+=1
        bs_all.append(tmp)
    if save_allpix: fout.close()
    return np.array(bs_all,dtype='complex')

def calc_cps_multi_DFT(ims,cdir,display=True,nx=256,ny=256,useW=True,save_allpix=False,filename='',redo_calc=False): #Master function
    """
    calculates closure phases using multiple pixel triangles
    for each triangle of baselines
    """

    dmats = []
    for i in range(35): ###remove hardcoding later!!!!!
        cpDFTmat_Re,cpDFTmat_Im = pyfits.getdata(cdir+'cpDFTmat_multi_'+str(nx)+'_'+str(i)+'.fits')
        IM2CT = cpDFTmat_Re + cpDFTmat_Im*1.0j
        dmats.append(IM2CT)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    imcount=0
    bs_all = []
    for i in tqdm(ims):
        FT = fft_image(i,nx,ny)
        bs = []
        tmp2 = []
        for ind in range(len(dmats)):
            ctphis = np.dot(dmats[ind],i.flatten())
            bispecs = []
            for j in range(len(ctphis)//3):
                bispec = np.prod([ctphis[j*3+x] for x in range(3)])
                bispecs.append(bispec)      
            bis_tmp = np.mean(bispecs)
            if save_allpix: 
                fout['int'+str(imcount)+'/tri'+str(ind)] = bispecs ###all pixel triangles for one triangle of baselines
            bs.append(bis_tmp)
        bs_all.append(bs)        
        imcount+=1
    cas = np.angle(bs_all,deg=1)
    Aas = np.abs(bs_all)
    Tas = np.abs(np.mean(bs_all,axis=0))
    cps = np.angle(np.mean(bs_all,axis=0),deg=1)
    if save_allpix: fout.close()
    if len(ims)>1: cov,var,stdE = gen_cov(cps,cas,W=Aas,useW=useW) 
    else: cov,var,stdE = None,None,None
    return np.array([bs_all,cps,Tas,cov,var,stdE])

def gen_cov(mean,data,W=None,useW=True):
    """
    generates covariance matrices with and without weights
    """
    ds = data - mean
    if useW==True:
        cov_num = (1.0/(len(ds)-1))*\
                np.array([[np.sum(np.array([W[im,x]*W[im,y]*ds[im,x]*ds[im,y] for im in range(len(ds))]))
                for x in range(len(mean))] for y in range(len(mean))])
        cov_den = np.array([[np.sum(np.array([W[im,x]*W[im,y] for im in range(len(ds))]))
                for x in range(len(mean))] for y in range(len(mean))])
        cov = cov_num / cov_den
    else:
        cov = (1.0/(len(ds)-1))*\
                np.array([[np.sum(np.array([ds[im,x]*ds[im,y] for im in range(len(ds))]))
                for x in range(len(mean))] for y in range(len(mean))])
    var = np.diag(cov)
    stdE = np.sqrt(np.diag(cov))*(1.0/np.sqrt(len(ds)))
    return cov,var,stdE

def mask_sig_pspec(mdir,nx,ny):
    """
    First attempt at calculating a bias for the visibilities...
    This masks out the splodges in the power spectra
    and returns an array where the non-splodge pixels are 0.
    """
    bcs = np.array(find_v2_pix(mdir))
    bcs2 = []
    for x in bcs:
        for y in x:
            bcs2.append(y)
    bcs2 = np.array(bcs2)
    ymin = int(ny/2+np.min(bcs2[:,1]-ny/2)*1.1)
    ymax = int(ny/2+np.max(bcs2[:,1]-ny/2)*1.1)
    xmin = int(nx/2+np.min(bcs2[:,0]-nx/2)*1.1)
    xmax = int(nx/2+np.max(bcs2[:,0]-nx/2)*1.1)

    if ymin < 0: ymin=0
    if xmin < 0: xmin=0
    if ymax > ny: ymax=ny
    if xmax > nx: xmax=nx
    m = np.zeros([ny,nx])
    m[ymin:ymax,xmin:xmax] = 1.0
    return m


def calc_vis_bias(psp,m,pix):
    """
    Calculates the mean in a single power spectrum that's not
    in the splodges. Multiplies by the number of pixels in
    each squared visibility to account for the bias.
    """
    vb = np.mean(psp[np.where(m==0)])
    vsum = vb*len(pix)
    return vsum

def make_blens(mdir):
    """
    calculates baseline lengths from uv sampling coordinates
    for plotting purposes
    """
    buvs = np.array(pyfits.getdata(mdir+'bl_uvs.fits'))
    blens = np.sqrt(buvs[:,0]**2+buvs[:,1]**2)
    return blens

def calc_v2s(ims,mdir,nx=256,ny=256,display=False,save_allpix=True,filename=None):
    """
    calculates squared visibilities for a stack of images
    """
    bcs = find_v2_pix(mdir)
    bcs = np.array(bcs)
    m=mask_sig_pspec(mdir,nx,ny)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    imcount=0
    v2s = []
    v2sn = []
    amps = []
    vun = []
    vbs = []
    pall = np.zeros([ny,nx])
    for i in tqdm(ims):
        psp = make_pspec(i,nx,ny)
        pall+=psp
        tmp = np.array([np.sum([psp[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))])
                        for x in range(len(bcs))])
        tmpall = np.array([[psp[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))]
                        for x in range(len(bcs))])
        v2s.append(tmp/psp[ny//2,nx//2])
        vun.append(tmp)
        amps.append(psp[ny//2,nx//2])
        bcs = np.array(bcs)
        vbs.append(np.array([
            calc_vis_bias(psp,m,bcs[x])/psp[ny//2,nx//2]
            for x in range(len(bcs))]))
        if save_allpix==True:
            for bb in range(len(tmpall)):
                fout['int'+str(imcount)+'/v2s'+str(bb)] = tmpall[bb] ###all pixels for one baseline
            fout['int'+str(imcount)+'/bias']=vbs[-1] ####bias values to subtract for each baseline
            fout['int'+str(imcount)+'/zsp'] = psp[ny//2,nx//2]
        imcount+=1
        if display==True and imcount==1:
            fff = plt.figure(figsize=(5,5))
            #plt.subplots_adjust(right = 0.99,left = 0.02,bottom=0.04, top = 0.95)
            #plt.title(mdir)
            ax = fff.add_subplot(111)
            plt.imshow(pall**0.1,origin='lower')
            #for x in range(len(bcs)):
            for x in [0]:
                plt.scatter(bcs[x][:,0],bcs[x][:,1],edgecolors='k',facecolors='none')
            ax.set_yticks([])
            ax.set_xticks([])
            #plt.xlim(20,237)
            #plt.ylim(20,237)
            #plt.savefig('/Users/stephsallum/Dropbox/Talks/220719_SPIE/v2_sampling.pdf')
            plt.show()
        #stop
        
    v2sc = np.array(v2s) - np.array(vbs)
    v2m = np.mean(v2sc,axis=0)
    if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    else: cov,var,stdE = None,None,None
    if save_allpix: fout.close()
    return np.array([v2m, cov, var, stdE, v2sc, amps, vun, vbs])

def calc_v2s_groups(ims,mdir,nx=256,ny=256,display=False,save_allpix=True,filename=None):
    """
    calculates squared visibilities for a stack of images
    """
    bcs = find_v2_pix(mdir)
    bcs = np.array(bcs)
    m=mask_sig_pspec(mdir,nx,ny)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    imcount=0
    v2s = []
    v2sn = []
    amps = []
    vun = []
    vbs = []
    pall = np.zeros([ny,nx])
    for cube in tqdm(ims):
        groupcount=0
        v2stmp = []
        v2sntmp = []
        ampstmp = []
        vuntmp = []
        vbstmp = []
        for i in cube:
            psp = make_pspec(i,nx,ny)
            pall+=psp
            tmp = np.array([np.sum([psp[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))])
                            for x in range(len(bcs))])
            tmpall = np.array([[psp[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))]
                            for x in range(len(bcs))])
            v2stmp.append(tmp/psp[ny//2,nx//2])
            vuntmp.append(tmp)
            ampstmp.append(psp[ny//2,nx//2])
            bcs = np.array(bcs)
            vbstmp.append(np.array([
                calc_vis_bias(psp,m,bcs[x])/psp[ny//2,nx//2]
                for x in range(len(bcs))]))
            if save_allpix==True:
                for bb in range(len(tmpall)):
                    fout['int'+str(imcount)+'/group'+str(groupcount)+'/v2s'+str(bb)] = tmpall[bb] ###all pixels for one baseline
                fout['int'+str(imcount)+'/group'+str(groupcount)+'/bias']=vbstmp[-1] ####bias values to subtract for each baseline
                fout['int'+str(imcount)+'/group'+str(groupcount)+'/zsp'] = psp[ny//2,nx//2]
            groupcount+=1
        imcount+=1
        v2s.append(v2stmp)
        v2sn.append(v2sntmp)
        amps.append(ampstmp)
        vun.append(vuntmp)
        vbs.append(vbstmp)
        if display==True and imcount==1:
            fff = plt.figure(figsize=(5,5))
            #plt.subplots_adjust(right = 0.99,left = 0.02,bottom=0.04, top = 0.95)
            #plt.title(mdir)
            ax = fff.add_subplot(111)
            plt.imshow(pall**0.1,origin='lower')
            #for x in range(len(bcs)):
            for x in [0]:
                plt.scatter(bcs[x][:,0],bcs[x][:,1],edgecolors='k',facecolors='none')
            ax.set_yticks([])
            ax.set_xticks([])
            #plt.xlim(20,237)
            #plt.ylim(20,237)
            #plt.savefig('/Users/stephsallum/Dropbox/Talks/220719_SPIE/v2_sampling.pdf')
            plt.show()
        #stop
        
    v2sc = np.array(v2s) - np.array(vbs)
    v2m = np.mean(v2sc,axis=0)
    if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    else: cov,var,stdE = None,None,None
    if save_allpix: fout.close()
    return np.array([v2m, cov, var, stdE, v2sc, amps, vun, vbs])

def calc_v2s_single(ims,mdir,nx=256,ny=256,display=False):
    """
    calculates squared visibilities for a stack of images
    """
    bcs = find_v2_pix(mdir,v2type='single')
    bcs = np.array(bcs)
    m=mask_sig_pspec(mdir,nx,ny)
    v2s = []
    v2sn = []
    amps = []
    vun = []
    vbs = []
    pall = np.zeros([ny,nx])
    for i in tqdm(ims):
        psp = make_pspec(i,nx,ny)
        pall+=psp
        tmp = np.array([np.sum([psp[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))])
                        for x in range(len(bcs))])
        tmpall = np.array([[psp[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))]
                        for x in range(len(bcs))])
        v2s.append(tmp/psp[int(len(psp)/2),int(len(psp)/2)])
        vun.append(tmp)
        amps.append(psp[int(len(psp)/2),int(len(psp)/2)])
        bcs = np.array(bcs)
        vbs.append(np.array([
            calc_vis_bias(psp,m,bcs[x])/psp[int(len(psp)/2),int(len(psp)/2)]
            for x in range(len(bcs))]))
    if display==True:
        fff = plt.figure(figsize=(18,9))
        plt.subplots_adjust(right = 0.99,left = 0.02,bottom=0.04, top = 0.95)
        plt.title(mdir)
        plt.imshow(pall**0.1,origin='lower')
        for x in range(len(bcs)):
            plt.scatter(bcs[x][:,0],bcs[x][:,1],edgecolors='k',facecolors='none')
        plt.show()
    v2sc = np.array(v2s)# - np.array(vbs)
    v2m = np.mean(v2sc,axis=0)
    if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    else: cov,var,stdE = None,None,None
    return np.array([v2m, cov, var, stdE, v2sc, amps, vun, vbs])

def calc_cvis(ims,mdir,nx=256,ny=256,display=False,save_allpix=False,filename='',
              subpixel=False,write_FTs=False):
    """
    calculates visibility amplitudes and phases for a stack of images
    """
    bcs = find_cvis_pix(mdir)
    bcs = np.array(bcs)
    m=mask_sig_pspec(mdir,nx,ny)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    vcs = []
    vbs = []
    pall = np.zeros([ny,nx])
    imcount=0
    for i in tqdm(ims):
        
        if subpixel==False: FT = fft_image(i,nx,ny)
        else: 
            yint,xint = get_center(i,4.3,6.5,0.065)
            x,y = find_psf_center(i,verbose=False)
            dy,dx = y-yint,x-xint
            FTo,FTn = fourier_center(i,dy,dx)
            img = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(FTn))))
            """
            plt.imshow(i)
            plt.colorbar()
            plt.scatter(x,y,c='r')
            plt.scatter(xint,yint,c='b')
            plt.show()
            plt.imshow(img)
            plt.colorbar()
            plt.show()
            plt.imshow(i-img)
            plt.colorbar()
            plt.show()
            stop
            """
            FT = fft_image(img,nx,ny)
        
        
        pall+=np.abs(FT)**2
        tmp = np.array([np.mean([FT[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))])
                        for x in range(len(bcs))])
        tmpall = np.array([[FT[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))]
                        for x in range(len(bcs))])
        vis = tmp/FT[ny//2,nx//2]
        vcs.append(vis)
        bcs = np.array(bcs)
        vbs.append(np.array([
            calc_vis_bias(np.abs(FT),m,bcs[x])/np.abs(FT)[ny//2,nx//2]
            for x in range(len(bcs))]))
        if save_allpix==True:
            for bb in range(len(tmpall)):
                fout['int'+str(imcount)+'/cvis'+str(bb)] = tmpall[bb] ###all pixels for one baseline
            fout['int'+str(imcount)+'/bias']=vbs[-1] ####bias values to subtract for each baseline
            fout['int'+str(imcount)+'/zsp'] = FT[ny//2,nx//2]
        if write_FTs==True:
            fout['int'+str(imcount)+'/FT']=FT
        imcount+=1
        #if imcount==9:
        #    FTt = FT
    if display==True:
        fff = plt.figure(figsize=(18,9))
        plt.subplots_adjust(right = 0.99,left = 0.02,bottom=0.04, top = 0.95)
        plt.title(mdir)
        #plt.imshow(pall**0.1,origin='lower')
        cpl = plt.imshow(np.angle(FT,deg=1),origin='lower')
        plt.colorbar(cpl)
        for x in range(len(bcs)):
            plt.scatter(bcs[x][:,0],bcs[x][:,1],edgecolors='k',facecolors='none')
        plt.show()
    amps_c = np.abs(np.array(vcs))# - np.array(vbs)
    amps_m = np.mean(amps_c,axis=0)
    phis_m = np.angle(np.mean(vcs,axis=0),deg=1)
    phis_all = np.angle(vcs,deg=1)
    #if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    #else: cov,var,stdE = None,None,None
    cov,var,stdE = None,None,None
    if save_allpix: fout.close()
    return np.array([amps_m, phis_m, cov, var, stdE, phis_all])

def calc_cvis_groups(ims,mdir,nx=256,ny=256,display=False,save_allpix=False,filename='',subpixel=False):
    """
    calculates visibility amplitudes and phases for a stack of images
    """
    bcs = find_cvis_pix(mdir)
    bcs = np.array(bcs)
    m=mask_sig_pspec(mdir,nx,ny)
    if save_allpix:
        fout = h5py.File(filename+'.hdf5', 'w')
    vcs = []
    vbs = []
    pall = np.zeros([ny,nx])
    imcount=0
    for cube in tqdm(ims):
        vcstmp = []
        vbstmp = []
        groupcount=0
        for i in cube:

            FT = fft_image(i,nx,ny)
            pall+=np.abs(FT)**2
            tmp = np.array([np.mean([FT[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))])
                            for x in range(len(bcs))])
            tmpall = np.array([[FT[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))]
                            for x in range(len(bcs))])
            vis = tmp/FT[ny//2,nx//2]
            vcstmp.append(vis)
            bcs = np.array(bcs)
            vbstmp.append(np.array([
                calc_vis_bias(np.abs(FT),m,bcs[x])/np.abs(FT)[ny//2,nx//2]
                for x in range(len(bcs))]))
            if save_allpix==True:
                for bb in range(len(tmpall)):
                    fout['int'+str(imcount)+'/group'+str(groupcount)+'/cvis'+str(bb)] = tmpall[bb] ###all pixels for one baseline
                fout['int'+str(imcount)+'/group'+str(groupcount)+'/bias']=vbstmp[-1] ####bias values to subtract for each baseline
                fout['int'+str(imcount)+'/group'+str(groupcount)+'/zsp'] = FT[ny//2,nx//2]
            groupcount+=1
        vcs.append(vcstmp)
        vbs.append(vbstmp)
        imcount+=1
    if display==True:
        fff = plt.figure(figsize=(18,9))
        plt.subplots_adjust(right = 0.99,left = 0.02,bottom=0.04, top = 0.95)
        plt.title(mdir)
        plt.imshow(pall**0.1,origin='lower')
        for x in range(len(bcs)):
            plt.scatter(bcs[x][:,0],bcs[x][:,1],edgecolors='k',facecolors='none')
        plt.show()
    amps_c = np.abs(np.array(vcs))# - np.array(vbs)
    amps_m = np.mean(amps_c,axis=0)
    phis_m = np.angle(np.mean(vcs,axis=0),deg=1)
    phis_all = np.angle(vcs,deg=1)
    #if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    #else: cov,var,stdE = None,None,None
    cov,var,stdE = None,None,None
    if save_allpix: fout.close()
    return np.array([amps_m, phis_m, cov, var, stdE, phis_all, vcs, vbs])
