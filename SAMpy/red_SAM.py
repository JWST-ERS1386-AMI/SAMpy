import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import copy
from scipy import ndimage
from jwst import datamodels
from jwst.datamodels import dqflags


def supergauss_fracw(pix,frac,m,size):
    sigma = (-float(pix)**float(m)/np.log(frac))**(1.0/float(m))
    ds = np.array([[np.sqrt(np.array((x-size/2)**2+(y-size/2)**2)) for x in range(size)]
        for y in range(size)])
    arr = np.exp(-ds**m/sigma**m)
    return arr

def apply_wind(ims,lam,ps,display=False):
    ims_wind = []
    for im in ims:
        size = len(im)
        wwidth = lam*1.0e-6*206265./ps*0.65
        hbox = supergauss_fracw(wwidth,0.95,4.0,size)
        ims_wind.append(im*hbox)
    if display==True:
        plt.imshow(hbox)
        plt.colorbar()
        plt.show()
    return np.array(ims_wind)


def read_calints(file):
    ff = pyfits.open(file)
    ims = ff[1].data
    bps = ff[3].data
    hdr0 = ff[0].header
    hdr1 = ff[1].header
    roll = hdr1['ROLL_REF']
    filt = hdr0['FILTER']
    ut_st = hdr0['EXPSTART']
    ut_end = hdr0['EXPEND']
    ut_m = hdr0['EXPMID']
    tfr = hdr0['TFRAME']
    foc = hdr0['FOCUSPOS']
    input_model = datamodels.CubeModel(file) ##add something so that it can also do Image Models
    dqs = input_model.dq
    bpmaps = np.zeros(ims.shape)
    print(dqflags.pixel)
    print(dqflags.group)
    #flaglist = ['DO_NOT_USE','SATURATED','JUMP_DET','DROPOUT','OUTLIER','AD_FLOOR',
    #            'DEAD','HOT','WARM','NONLINEAR']
    flaglist = ['DO_NOT_USE']
    for flag in flaglist:
        bpmaps[np.where(input_model.dq==dqflags.pixel[flag])]=1.0
    #bpmaps[np.where(np.isin(input_model.dq,list(dqflags.pixel.values())) == False)]=1.0
    return ims,dqs,bpmaps,roll,ut_m,filt

def read_cal(file):
    ff = pyfits.open(file)
    ims = ff[1].data
    bps = ff[3].data
    hdr0 = ff[0].header
    hdr1 = ff[1].header
    roll = hdr1['ROLL_REF']
    ut_st = hdr0['EXPSTART']
    ut_end = hdr0['EXPEND']
    ut_m = hdr0['EXPMID']
    tfr = hdr0['TFRAME']
    filt = hdr0['FILTER']
    foc = hdr0['FOCUSPOS']
    input_model = datamodels.ImageModel(file) ##add something so that it can also do Image Models
    dqs = input_model.dq
    bpmaps = np.zeros(ims.shape)
    #flaglist = ['DO_NOT_USE','SATURATED','JUMP_DET','DROPOUT','OUTLIER','AD_FLOOR',
    #            'DEAD','HOT','WARM','NONLINEAR']
    flaglist = ['DO_NOT_USE']
    for flag in flaglist:
        bpmaps[np.where(input_model.dq==dqflags.pixel[flag])]=1.0
    #bpmaps[np.where(np.isin(input_model.dq,list(dqflags.pixel.values())) == False)]=1.0
    return ims,dqs,bpmaps,roll,ut_m,filt

def read_calims(filelist,indir=''):
    ims = []
    dqs = []
    bpmaps = []
    rolls = []
    uts_m = []
    filts = []
    for file in filelist:
        print(file)
        if file[-5:]=='.fits':
            im,dqim,bpmap,roll,ut_m,filt = read_cal(indir+file)
            ims.append(im)
            dqs.append(dqim)
            bpmaps.append(bpmap)
            rolls.append(roll)
            uts_m.append(ut_m)
            filts.append(filt)
    return np.array(ims),np.array(dqs),np.array(bpmaps),np.array(rolls),np.array(uts_m),filts



def bkg_sub(ims,rmin=40,rmax=50):
    nfr,ny,nx = ims.shape
    print(ims.shape)
    dfs = np.array([[np.sqrt((x-nx/2.0)**2+(y-ny/2.0)**2)
                    for x in range(nx)]
                    for y in range(ny)])
    imtest = np.zeros(ims[0].shape)
    imtest[np.where((dfs < rmax) & (dfs > rmin))] = 1.0
    ims_b = []
    for im in ims:
        imb = copy.deepcopy(im)
        tomean = imb[np.where((dfs < rmax) & (dfs > rmin))]
        mbkg = np.mean(tomean)
        imb-=mbkg
        ims_b.append(imb)
    plt.imshow(im**0.1)
    plt.contour(imtest,levels=[1],colors='w')
    plt.show()
    return np.array(ims_b)


def badpixels(ims,bp=[],size=1,display=False):
    """
    corrects image for bad pixels using a bad pixel map
        -subframes bad pixel map to isize by isize
        -takes mean of good pixels in size by size box around each bad pixel
        -replaces bad pixels with that mean, marks them as good
    """
    if bp==[]:
        return ims
    imsb = copy.deepcopy(ims)
    for ii in range(len(imsb)):
        bpcopy = copy.deepcopy(bp[ii])
        ps = np.where(bpcopy!=0)
        ind = 0
        while False in np.unique(bpcopy==0):
            #print(ps)
            y,x = ps[0][ind],ps[1][ind]
            bottom = y < size
            top = y > len(bp)-size
            left = x < size
            right = x > len(bp[0])-size
            bl = y-size
            tl = y+size+1
            ll = x-size
            rl = x+size+1
            if bottom: bl=0
            if top: tl=len(bp[ii])
            if right: rl=len(bp[ii][0])
            if left: ll=0
            tomean = ims[ii,bl:tl,ll:rl][np.where(bpcopy[bl:tl,ll:rl] == 0.0)]
            #print(ii,x,y,bl,tl,ll,rl,bottom,top,left,right)
            #print(tomean)

            if len(tomean) > 0:
                imsb[ii,y,x] = np.median(tomean)
                bpcopy[y,x] = 0
                ps = np.where(bpcopy!=0)
                ind=0
            else:
                ind+=1
    if display==True:
        pind = 0
        f = plt.figure(figsize=(12,6))
        f.add_subplot(131)
        plt.title('Input')
        plt.imshow(ims[pind]**0.1)
        f.add_subplot(132)
        plt.title('Bad Pixel Map')
        plt.imshow(bp[pind]**0.1)
        f.add_subplot(133)
        plt.title('Output')
        plt.imshow(imsb[pind]**0.1)
        plt.show()
    return imsb
    
def center_interf(image,size,display=False):
    """
         uniform filter image
         then center interferogram
    """
    im1f=ndimage.filters.gaussian_filter(image,size)
    y,x=np.unravel_index(np.argmax(im1f.flatten()),im1f.shape)
    if display == True:
        f = plt.figure()
        plt.imshow(im1f,origin='lower')
        plt.scatter(x,y)
        plt.show()
        var = input('are centers correct? (y/n)')
        if var == 'n':
            y = int(raw_input('enter y coord: '))
            x = int(raw_input('enter x coord: '))
    return y,x


def subframe(ims,sfsize=70,sm=5):
    ims_s = []
    for im in ims:
        y,x = center_interf(im,sm)
        imsub = im[y-sfsize//2:y+sfsize//2,
                   x-sfsize//2:x+sfsize//2]
        ims_s.append(imsub)
    return np.array(ims_s)



    
    
    
    
