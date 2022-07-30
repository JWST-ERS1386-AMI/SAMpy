from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import map_coordinates
from supergauss_new import supergauss_fracw
import cmath
import math
import os
import numpy as np
import astropy.io.fits as pyfits
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy import interpolate
import pdb





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

def find_tris_multi(ccs,cdir,ny=256,nx=256): #Function 2 definition
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
        if os.path.isfile(ifile)==False: #Executed during first run
            print ('\n Calculating triangle: '+str(x+1)+' of '+str(len(ccs)))
            ib = cdir+'ind'+str(x)+'_vert'
            p0 = pyfits.getdata(ib+'0.fits')
            p1 = pyfits.getdata(ib+'1.fits')
            p2 = pyfits.getdata(ib+'2.fits')
            g = []
            for pp1 in tqdm(p0):
                for pp2 in p1:
                    dpp1 = pp1 - np.array([int(ny/2),int(nx/2)])
                    dpp2 = pp2 - np.array([int(ny/2),int(nx/2)])
                    dpp3 = -dpp1-dpp2
                    pp3 = dpp3 + np.array([int(ny/2),int(nx/2)])
                    t1 = np.where(p2[:,0]==pp3[0])
                    t2 = np.where(p2[:,1]==pp3[1])
                    inlist = False
                    for t in t1[0]:
                        if t in t2[0]: inlist=True
                    if inlist==True:
                        g.append([pp1,pp2,pp3])
            g = np.array(g)
            pyfits.writeto(ifile,g)
        else: g = pyfits.getdata(ifile)
        ccall.append(g)
    return np.array(ccall)

def make_blens(mdir):
    """
    calculates baseline lengths from uv sampling coordinates
    for plotting purposes
    """
    buvs = np.array(pyfits.getdata(mdir+'bl_uvs.fits'))
    blens = np.sqrt(buvs[:,0]**2+buvs[:,1]**2)
    return blens

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
                plt.show()
        imcount+=1
    cas = np.angle(bs_all,deg=1)
    Aas = np.abs(bs_all)
    cps = np.angle(np.mean(bs_all,axis=0),deg=1)
    cov,var,stdE = gen_cov(cps,cas,W=Aas,useW=useW)
    return bs_all, cps, cov, var, stdE

def calc_cps_multi(ims,cdir,display=True,nx=256,ny=256,useW=True): #Master function
    """
    calculates closure phases using multiple pixel triangles
    for each triangle of baselines
    """
    ccs = make_cpcoords(cdir)
    gcs = find_tris_multi(ccs,cdir)
    imcount=0
    bs_all = []
    for i in tqdm(ims):
        FT = fft_image(i,nx,ny)
        bs = []
        for ind in range(len(gcs)):
            gc = gcs[ind]
            if display==True:
                if imcount==0:
                    f=plt.figure(figsize=(10,10))
                    plt.imshow(np.abs(FT)**0.1,cmap='viridis')#,vmin=0, vmax=0.4
                    plt.imshow(np.abs(FT)**0.1)
                    tp = np.mean(gc,axis=0)
                    plt.scatter(gc[:,:,0],gc[:,:,1],edgecolors='k',facecolors='None')
                    plt.show()
            tomean = []
            for p in range(len(gc)):
                for j in range(3):
                    if j==0:
                        tomult = FT[gc[p,j,1],gc[p,j,0]]/FT[ny//2,nx//2]
                    else: tomult*=FT[gc[p,j,1],gc[p,j,0]]/FT[ny//2,nx//2]
                tomean.append(tomult)
            bis_tmp = np.mean(tomean)
            bs.append(bis_tmp)
        bs_all.append(bs)
        imcount+=1
    cas = np.angle(bs_all,deg=1)
    Aas = np.abs(bs_all)
    Tas = np.abs(np.mean(bs_all,axis=0))
    cps = np.angle(np.mean(bs_all,axis=0),deg=1)
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

def calc_v2s(ims,mdir,nx=256,ny=256,display=False):
    """
    calculates squared visibilities for a stack of images
    """
    bcs = find_v2_pix(mdir)
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
    v2sc = np.array(v2s) - np.array(vbs)
    v2m = np.mean(v2sc,axis=0)
    if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    else: cov,var,stdE = None,None,None
    return np.array([v2m, cov, var, stdE, v2sc, amps, vun, vbs])

def calc_cvis(ims,mdir,nx=256,ny=256,display=False):
    """
    calculates visibility amplitudes and phases for a stack of images
    """
    bcs = find_cvis_pix(mdir)
    bcs = np.array(bcs)
    m=mask_sig_pspec(mdir,nx,ny)
    vcs = []
    vbs = []
    pall = np.zeros([ny,nx])
    for i in tqdm(ims):
        FT = fft_image(i,nx,ny)
        pall+=np.abs(FT)**2
        tmp = np.array([np.mean([FT[bcs[x][y][1],bcs[x][y][0]] for y in range(len(bcs[x]))])
                        for x in range(len(bcs))])
        vis = tmp/FT[ny//2,nx//2]
        vcs.append(vis)
        bcs = np.array(bcs)
        vbs.append(np.array([
            calc_vis_bias(np.abs(FT),m,bcs[x])/np.abs(FT)[ny//2,nx//2]
            for x in range(len(bcs))]))
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
    if len(ims)>1: cov,var,stdE = gen_cov(v2m,v2sc,useW=False)
    else: cov,var,stdE = None,None,None
    return np.array([amps_m, phis_m, cov, var, stdE])
