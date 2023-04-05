import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import copy
from scipy import ndimage
from jwst import datamodels
from jwst.datamodels import dqflags
from ipywidgets import IntProgress
from IPython.display import display as idisplay


def supergauss_fracw(pix,frac,m,size):
    sigma = (-float(pix)**float(m)/np.log(frac))**(1.0/float(m))
    ds = np.array([[np.sqrt(np.array((x-size/2)**2+(y-size/2)**2)) for x in range(size)]
        for y in range(size)])
    arr = np.exp(-ds**m/sigma**m)
    return arr

def apply_wind(ims,lam,ps,display=False):
    ims_wind = []
    wwidth = lam*1.0e-6*206265./ps*0.65
    size = len(ims[0])
    hbox = supergauss_fracw(wwidth,0.95,4.0,size)
    for im in ims:
        ims_wind.append(im*hbox)
    if display==True:
        plt.imshow(hbox)
        plt.colorbar()
        plt.show()
    return np.array(ims_wind)


def apply_wind_groups(cubes,lam,ps,display=False):
    ims_wind = np.zeros(cubes.shape)
    wwidth = lam*1.0e-6*206265./ps*0.65
    size = len(cubes[0,0])
    hbox = supergauss_fracw(wwidth,0.95,4.0,size)
    for ii,ims in enumerate(cubes):
        for jj,im in enumerate(ims):
            ims_wind[ii,jj]=im*hbox
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
    roll_ref = hdr1['ROLL_REF']
    vpar = hdr1['VPARITY']
    v31_yang = hdr1['V3I_YANG']
    parang = roll_ref - v31_yang*vpar    
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
    #flaglist = ['DO_NOT_USE','JUMP_DET','SATURATED']
    flaglist = ['DO_NOT_USE']

    for flag in flaglist:
        bpmaps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
    #bpmaps[np.where(np.isin(input_model.dq,list(dqflags.pixel.values())) == False)]=1.0
    return ims,dqs,bpmaps,parang,ut_m,filt

def read_jumpstep_ims(filelist):
    ims_all = []
    dqs_all = []
    bpmaps_all = []
    rolls_all = []
    uts_m_all = []
    filts_all = []
    for file in filelist:
        ff = pyfits.open(file)
        ims = ff[1].data
        bps = ff[3].data
        hdr0 = ff[0].header
        hdr1 = ff[1].header
        roll_ref = hdr1['ROLL_REF']
        vpar = hdr1['VPARITY']
        v31_yang = hdr1['V3I_YANG']
        parang = roll_ref - v31_yang*vpar
        filt = hdr0['FILTER']
        ut_st = hdr0['EXPSTART']
        ut_end = hdr0['EXPEND']
        ut_m = hdr0['EXPMID']
        tfr = hdr0['TFRAME']
        foc = hdr0['FOCUSPOS']
        input_model = datamodels.QuadModel(file) ##add something so that it can also do Image Models
        dqs = input_model.groupdq
        bpmaps = np.zeros(ims.shape)
        #flaglist = ['DO_NOT_USE','SATURATED','JUMP_DET','DROPOUT','OUTLIER','AD_FLOOR',
        #            'DEAD','HOT','WARM','NONLINEAR']
        #flaglist = ['DO_NOT_USE','JUMP_DET','SATURATED']
        flaglist = ['DO_NOT_USE']

        for flag in flaglist:
            bpmaps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
        #bpmaps[np.where(np.isin(input_model.dq,list(dqflags.pixel.values())) == False)]=1.0
        for ii in range(len(ims)):
            ims_all.append(ims[ii])
            dqs_all.append(dqs[ii])
            bpmaps_all.append(bpmaps[ii])
            filts_all.append(filt)
            rolls_all.append(parang)
            uts_m_all.append(ut_m)
    return np.array(ims_all),np.array(dqs_all),np.array(bpmaps_all),np.array(rolls_all),np.array(uts_m_all),filts_all



def read_calint_ims(filelist):
    ims_all = []
    dqs_all = []
    bpmaps_all = []
    rolls_all = []
    uts_m_all = []
    filts_all = []
    for file in filelist:
        ff = pyfits.open(file)
        ims = ff[1].data
        bps = ff[3].data
        hdr0 = ff[0].header
        hdr1 = ff[1].header
        roll_ref = hdr1['ROLL_REF']
        vpar = hdr1['VPARITY']
        v31_yang = hdr1['V3I_YANG']
        parang = roll_ref - v31_yang*vpar
        filt = hdr0['FILTER']
        ut_st = hdr0['EXPSTART']
        ut_end = hdr0['EXPEND']
        ut_m = hdr0['EXPMID']
        tfr = hdr0['TFRAME']
        foc = hdr0['FOCUSPOS']
        input_model = datamodels.CubeModel(file) ##add something so that it can also do Image Models
        dqs = input_model.dq
        bpmaps = np.zeros(ims.shape)
        flaglist = ['DO_NOT_USE','SATURATED','JUMP_DET','DROPOUT','OUTLIER','AD_FLOOR',
                    'DEAD','HOT','WARM','NONLINEAR']
        #flaglist = ['DO_NOT_USE','JUMP_DET','SATURATED']
        #flaglist = ['DO_NOT_USE']

        for flag in flaglist:
            bpmaps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
        #bpmaps[np.where(np.isin(input_model.dq,list(dqflags.pixel.values())) == False)]=1.0
        for ii in range(len(ims)):
            ims_all.append(ims[ii])
            dqs_all.append(dqs[ii])
            bpmaps_all.append(bpmaps[ii])
            filts_all.append(filt)
            rolls_all.append(parang)
            uts_m_all.append(ut_m)
    return np.array(ims_all),np.array(dqs_all),np.array(bpmaps_all),np.array(rolls_all),np.array(uts_m_all),filts_all

def read_cal(file):
    ff = pyfits.open(file)
    ims = ff[1].data
    bps = ff[3].data
    hdr0 = ff[0].header
    hdr1 = ff[1].header
    roll_ref = hdr1['ROLL_REF']
    vpar = hdr1['VPARITY']
    v31_yang = hdr1['V3I_YANG']
    parang = roll_ref - v31_yang*vpar
    ut_st = hdr0['EXPSTART']
    ut_end = hdr0['EXPEND']
    ut_m = hdr0['EXPMID']
    tfr = hdr0['TFRAME']
    filt = hdr0['FILTER']
    foc = hdr0['FOCUSPOS']
    input_model = datamodels.ImageModel(file) ##add something so that it can also do Image Models
    dqs = input_model.dq
    bpmaps = np.zeros(ims.shape)
    #flaglist = ['DO_NOT_USE','SATURATED','DROPOUT','OUTLIER','PERSISTENCE']
    #flaglist = ['DO_NOT_USE','SATURATED','JUMP_DET','DROPOUT','OUTLIER','AD_FLOOR',
    #            'DEAD','HOT','WARM','NONLINEAR']
    #flaglist = ['DO_NOT_USE','SATURATED',
    #            'DEAD','HOT','WARM','NONLINEAR']
    #flaglist = ['DO_NOT_USE','JUMP_DET','SATURATED']
    flaglist = ['DO_NOT_USE']

    #print(dqflags.pixel)
    #stop
    for flag in flaglist:
        bpmaps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
    #bpmaps[np.where(np.isin(input_model.dq,list(dqflags.pixel.values())) == False)]=1.0
    return ims,dqs,bpmaps,parang,ut_m,filt

def parse_dqmap(file):
    input_model = datamodels.ImageModel(file)
    dqs = input_model.dq
    bpvals = list(dqflags.pixel.values())
    bpkeys = list(dqflags.pixel.keys())
    plist = []
    for y in range(len(dqs)):
        for x in range(len(dqs[y])):
            tmp = [y,x]
            for i,val in enumerate(bpvals):
                if dqs[y,x] & val > 0:
                    tmp.append(bpkeys[i])
                    #print(y,x,i,val,bpkeys[i])
            if len(tmp) > 2: 
                print(tmp)
                plist.append(tmp)
    return plist

def parse_dqmap_ints(file):
    input_model = datamodels.CubeModel(file)    
    dqs = input_model.dq
    bpvals = list(dqflags.pixel.values())
    bpkeys = list(dqflags.pixel.keys())
    fr = 0
    plist = []
    for y in range(len(dqs[fr])):
        for x in range(len(dqs[fr,y])):
            tmp = [y,x]
            for i,val in enumerate(bpvals):
                if dqs[fr,y,x] & val > 0:
                    tmp.append(bpkeys[i])
                    #print(y,x,i,val,bpkeys[i])
            if len(tmp) > 2: 
                print(tmp)
                plist.append(tmp)
    return plist

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
    print(ims.shape)
    print(bp.shape)
    print(imsb.shape)
    f = IntProgress(min=0, max=len(imsb)) # instantiate the bar
    idisplay(f) # display the bar
    f.value=0
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
        f.value+=1
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


def subframe(ims,sfsize=71,sm=5):
    dims = ims.shape
    if len(dims)==4:
        y,x = center_interf(np.nanmedian(np.nanmedian(ims,axis=0),axis=0),sm)
        ims_s = []
        for imcube in ims:
            tmp = []
            for im in imcube:
                imsub = im[y-sfsize//2:y+sfsize//2+1,
                       x-sfsize//2:x+sfsize//2+1]
                tmp.append(imsub)
            ims_s.append(tmp)
    else:
        ims_s = []
        y,x = center_interf(np.nanmedian(ims,axis=0),sm)
        print(y,x)
        for im in ims:
            #y,x = center_interf(im,sm)
            imsub = im[y-sfsize//2:y+sfsize//2+1,
                       x-sfsize//2:x+sfsize//2+1]
            ims_s.append(imsub)
    return np.array(ims_s)


def subframe_circ(ims,sfsize=69,sm=5):
    dims = ims.shape
    if len(dims)==4:
        medim = np.nanmedian(np.nanmedian(ims,axis=0),axis=0)
        y,x = center_interf(medim,sm)
        dists = np.array([[np.sqrt((ii-x)**2 + (jj-y)**2) 
                           for ii in range(len(medim))] 
                          for jj in range(len(medim))])
        ims_s = []
        for imcube in ims:
            tmp = []
            for im in imcube:
                imcopy = copy.deepcopy(im)
                imcopy[np.where(dists > sfsize//2)] = 0.0
                imsub = imcopy[y-sfsize//2:y+sfsize//2,
                       x-sfsize//2:x+sfsize//2]
                tmp.append(imsub)
            ims_s.append(tmp)
    else:
        ims_s = []
        medim = np.nanmedian(ims,axis=0)
        y,x = center_interf(medim,sm)
        dists = np.array([[np.sqrt((ii-x)**2 + (jj-y)**2) 
                           for ii in range(len(medim))] 
                          for jj in range(len(medim))])
        for im in ims:
            #y,x = center_interf(im,sm)
            imcopy = copy.deepcopy(im)
            imcopy[np.where(dists > sfsize//2)] = 0.0
            imsub = imcopy[y-sfsize//2:y+sfsize//2,
                       x-sfsize//2:x+sfsize//2]
            ims_s.append(imsub)
    return np.array(ims_s)





    
    
    
    
