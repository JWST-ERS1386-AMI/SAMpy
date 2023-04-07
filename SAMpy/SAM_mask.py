import os
from scipy import ndimage
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from tqdm import tqdm
from astropy.io import fits

def diag(Ts):
    '''
    Extracts a matrix with rows containing the (unitary) eigenvectors of a
    hermitian which have corresponding eigenvalues of >1e-6
    --Parameter(s)--
    Ts: numpy array (2D)
        -> Passed hermitian matrix
    --Return(s)--
    (S_eigenvectors).real: numpy array (2D)
        -> Real part of the matrix with rows containing the unitary
        eigenvectors (of Ts) which have corresponding eigenvalues of >1e-6
    '''
    #Returns eigenvalues and (unitary) eigenvectors respectively for hermitians
    eigenvalues,eigenvectors = np.linalg.eigh(Ts)
    S_eigenvectors=eigenvectors.T #Eigenvectors are in each coulumn by default
    for i in range(len(eigenvalues)):
        if eigenvalues[i]<1e-6:
            S_eigenvectors=np.delete(S_eigenvectors,i,0)
    return (S_eigenvectors).real


def make_coords(cdir,jwst_filt='f380m',inst='niriss',pscam=0.0656,npix=256,rot=0,xadj=0,yadj=0,pmask=125,sp=20,redo=False,fcut=0.5):
    """
    Generates the sampling coordinates. Desinged to achieve the same by
    performing the following steps:
        1. Loading the coordinates of the Non Redundant Mask (here: JWST/NIRISS)
        2. Rotating (by 'rot' degrees) and shifting (to centre at {'xadj',
        'yadj'}) ? the NRM mask and modifying the coordinates accordingly
        3. Calculating the closure phase and baseline vectors from the above
        4. Creating matrix 'mat' with elements as ±1 if the closure phase
        triangle vectors are parallel/antiparallel respectively to baseline
        vectors and as 0 otherwise
        
    --Parameter(s)--
    nholes: integer
        ->Number of holes in the mask (here: 7 for NRM of JWST/NIRISS)
    jwst_filt: string
        ->JWST/NIRISS AMI compatible filter, one of the following:
        'f227w','f380m','f430m','f480m'
    inst: string
        ->'niriss' or 'sphere' (not yet included)
    --Return(s)--
    void
    """


    if (jwst_filt=='f227w'): TransmissionFile='NIRISS_F277W.txt'
    elif (jwst_filt=='f380m'):
        TransmissionFile='NIRISS_F380M.txt'
        """lowestindex =       3.682    0.0106376   0.00847809  /3182
        highestindex = 3.949     0.107237    0.0833171 /3451"""
    elif (jwst_filt=='f430m'): TransmissionFile='NIRISS_F430M.txt'
    elif (jwst_filt=='f480m'): TransmissionFile='NIRISS_F480M.txt'
    else: raise ValueError('jwst_filt must be a string with a JWST/NIRISS AMI\
     compatible filter and hence one of the following: \n f227w \n f380m\
      \n f430m \n f480m')


    if inst == 'niriss':
        nholes=7
        #in fourier space for the power spectrum
        #fcut = 0.3
        sd_m=0.75
        #mname='/Users/sray/Desktop/WorkFile/Year2/AMIPipeline/Sallum/Analysis/NIRISS_7holeMask.txt'
        mname = 'NIRISS_7holeMask.txt'
        
    """This part reads the coordinates of the mask"""
    if os.path.isdir(cdir)==False: os.mkdir(cdir)
    post = np.loadtxt(mname)
    post[np.where(post[:,0] > 0)] += np.array([xadj,yadj])
    c,s = np.cos(np.radians(-rot)),np.sin(np.radians(-rot))
    pos = np.array([[p[0]*c-p[1]*s,p[0]*s+p[1]*c] for p in post])
    #pos[:,0] = -1.0*pos[:,0]
    
    """Calculating closure phase vectors"""
    cps = np.array([[pos[x]-pos[y],pos[y]-pos[z],pos[z]-pos[x]]
        for x in range(len(pos)) for y in range(len(pos))
            for z in range(len(pos)) if z>y>x])

    """Calculating baseline vectors"""
    bls = np.array([pos[x]-pos[y] for x in range(len(pos))
        for y in range(len(pos)) if y>x])



    """Matrix to assign elements as ±1 if the closure phase triangle vectors
    are parallel/antiparallel respectively to baseline vectors and
    as 0 otherwise"""
    mat = []
    for tri in cps:
        ent = np.zeros(len(bls))
        for v in tri:
            for b in range(len(bls)):
                if (v[0] == bls[b,0]) and (v[1]==bls[b,1]): ent[b] = 1.0
                if (v[0] == -bls[b,0]) and (v[1]==-bls[b,1]): ent[b] = -1.0
        mat.append(ent)
        np.savetxt(cdir+'k_mat.txt',np.array(mat))

    bh = np.array([[pos[x],pos[y]]
            for x in range(len(pos))
            for y in range(len(pos))
            if y > x]) #3d Array with elements having start and end coordinates of all unique baselines


    bls_uv=np.matrix([row[:] for row in bls]) #Copying bls
    bls_uv[:,0]*=-1 #recording u as +ve going left in image
    pyfits.writeto(cdir+'bl_uvs.fits',bls_uv,overwrite=True)
    bl_arr = np.zeros([len(bls_uv),npix,npix])


    #Loading transmission profile of specific filter
    TransmissionFileData=np.loadtxt(TransmissionFile, skiprows=1)
    #Storing the wavelngth and transmission factor in lams and trans respcetively
    lams,trans = TransmissionFileData[:,0],TransmissionFileData[:,1]
    #Selecting values which have a high transmission
    lams,trans=lams[np.where(trans>0.5)],trans[np.where(trans>0.5)]
    #Selecting every nth point
    lams,trans=lams[::sp],trans[::sp]



    lamc = np.sum(lams*trans)/np.sum(trans) #Central wavelength
    lamm = [np.min(lams),np.max(lams)] #Wavelength boundary ? [::sint] ?

    psc = 1.0/(float(npix)*pscam)*206265.0*lamc*1e-06
    

    bls_pix = np.round(np.array([npix//2,npix//2]))+bls/psc #in pixel coordinates?
    pyfits.writeto(cdir+'bl_pix.fits',bls_pix,overwrite=True)
    bsm = np.max([sd_m/(1.0/(float(npix)*pscam)*206265.0*j*1e-06) for j in lamm])





    bl_pix_arr=[]
    PSall = np.zeros([npix,npix])
    fg_u = np.array([[[y,x]
                         for x in range(npix)]
                         for y in range(npix)])

    for i in range(len(bls_uv)):
        if ((os.path.isfile(cdir+'v2_ind'+str(i)+'.fits')==False) or (redo==True)):
            ftmp = np.zeros([npix,npix]) #fourier transform map

            print('Doing baseline',i+1,'of',len(bls_uv),'->')

            for ll in tqdm(range(len(lams))): #loading transmission profile
                lam = lams[ll]
                tt = trans[ll]

                #print('Doing wavelength=',lam)

                """Make NRM mask here"""
                psc_des = 0.025
                
                npixFT=int(np.round(1.0/(psc_des/(206265.0*lam*1e-06)*pscam)))
                if npixFT%2==0: npixFT+=1
                if i==0 and ll==0:
                    Fall = np.zeros([npixFT,npixFT])
                
                #print(npixFT)
                
                fg = np.array([[[y,x]
                         for x in range(npixFT)]
                         for y in range(npixFT)]) 

                f = np.zeros([npixFT,npixFT])
                #ps=1/(npix*pscam)*206265*lam*1e-06
                #bh_p is a coordinate change of bh according to the fourier plane and takes 2 points at a time
                bh_p = np.array(np.round(np.array([npixFT//2,npixFT//2])
                + bh[i]/psc_des),dtype=int)

                #print('bh_p=',bh_p)

                if inst=='vlt': #circular holes
                    sd = sd_m / ps #subaperture diametre adjusted to platescale
                    for p in bh_p:
                        ds = np.sqrt(np.sum((fg - [p[1],p[0]])**2,axis=2))
                        f[np.where(ds <= sd)] = 1.0


                elif inst=='niriss': #hexagonal holes
                    sd_m=0.75 #flat to flat distance of subaperture (hexagonal)
                    sd = sd_m / psc_des #flat to flat distance adjusted to platescale
                    for p in bh_p: #for each hole pair
                        HoleInPlane=[] #Mapping invididual holes
                        hole_Centre=p[1],p[0]
                        hole_SideLength=sd/(math.sqrt(3))
                        vertices=[] #vertices of hexagonal holes
                        for new_index in range (6): # finding vertices of hexagonal hole
                            RotationAngle=np.radians(new_index*60)
                            vertex_x, vertex_y = \
                            hole_Centre[0]+hole_SideLength*np.sin(RotationAngle),\
                            hole_Centre[1]+hole_SideLength*np.cos(RotationAngle)
                            vertex=np.array([vertex_x, vertex_y])
                            vertices.append(vertex)
                        hole_hexagon = Polygon(np.array(vertices))

                        for PixelLine in fg: #changing values at the hole: 0->1
                            #1d slice of the plane to map the holes
                            HoleInPlane_slice=[]
                            for Pixel in PixelLine:
                                #print('PixelHole=',Pixel)
                                point = Point(Pixel)
                                #Light passes
                                if hole_hexagon.contains(point):
                                    HoleInPlane_slice.append(1)
                                #Light doesn't pass
                                else: HoleInPlane_slice.append(0)
                            HoleInPlane.append(np.array(HoleInPlane_slice))
                        f=f+HoleInPlane #combining the holes one by one
                        
                    #obtaining the transpose to match the published coordinates
                    if ll==0:
                        plt.imshow(f,origin='lower')
                        plt.show()

                else:
                    raise ValueError('inst must be a string specifying one of the\
                    these instruments: niriss (for JWST/NIRISS),\
                     vlt (for SPHERE/VLT)')

                FT = abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f))))**2
                #plt.imshow(FT[npixFT//2-25:npixFT//2+26,npixFT//2-25:npixFT//2+26]**0.1)
                #plt.show()
                #stop
                if npix%2!=0:
                    ftmp += FT[npixFT//2-npix//2:npixFT//2+npix//2+1,npixFT//2-npix//2:npixFT//2+npix//2+1]*tt
                else:
                    ftmp += FT[npixFT//2-npix//2:npixFT//2+npix//2,npixFT//2-npix//2:npixFT//2+npix//2]*tt
                if ll==0: Fall+=np.transpose(f)
            PS = np.fft.fftshift(abs(np.fft.fft2(np.fft.fftshift(ftmp))))
            PS = PS/np.amax(PS) #normalising power spetrum to highest value
            
            #print('Masking the central maxima of the power spectrum')
            if inst=='vlt': #Circular masking
                ds = np.sqrt(np.sum((fg-[npix/2,npix/2])**2,axis=2))
                PS[np.where(ds<pmask)] = 0.0

            elif inst =='niriss': #Hexagonal masking
                MaskLayer = np.zeros(np.shape(PS)) #np.full((np.shape(PS)), 1)
                mask_Centre = (npix//2),(npix//2)
                mask_SideLength = int(125.0/4.0)/4.8*3.8*npix/257*1.0/(math.sqrt(3))
                vertices=[] #Vertices of hexagonal mask for power spectrum
                for mask_index in range (6): #finding vertices of hexagonal mask
                    RotationAngle=np.radians(mask_index*60)
                    vertex_x, vertex_y = \
                    mask_Centre[0]+mask_SideLength*np.sin(RotationAngle),\
                    mask_Centre[1]+mask_SideLength*np.cos(RotationAngle)
                    vertex=np.array([vertex_x, vertex_y])
                    vertices.append(vertex)
                mask_hexagon = Polygon(np.array(vertices))
                MaskInPlane=[]
                for PixelLine in fg_u: #changing values at the mask:
                    #1d slice of the plane to map the holes
                    MaskInPlane_slice=[]
                    for Pixel in PixelLine:
                        #print('MaskPixel=',Pixel)
                        point = Point(Pixel)
                        #Light passes
                        if mask_hexagon.contains(point):
                            MaskInPlane_slice.append(1)
                        #Light doesn't pass
                        else: MaskInPlane_slice.append(0)
                    MaskInPlane.append(np.array(MaskInPlane_slice))
                MaskLayer=MaskLayer+MaskInPlane #combining the holes one by one
                #Flipping the 0s to 1s and vice versa: 0s at central maxima
                MaskLayer=1-MaskLayer
                MaskLayer=np.transpose(MaskLayer) #to be consistent with the holes
                PS=np.multiply(MaskLayer,PS)
                if ll==0:
                    plt.imshow(PS**0.1)
                    plt.show()
            bl_arr[i]=PS
            PSall+=PS

            ss = np.where(bl_arr[i] > fcut)
            ####writing out x, y coords - these are the coordinates that will get sampled in the FTed images
            S = np.array([[ss[1][x],ss[0][x]] for x in range(len(ss[0]))])
            if len(S)==0:
                print ('nowhere to sample!')
                exit()
            pyfits.writeto(cdir+'v2_ind'+str(i)+'.fits',np.array(S),overwrite=True)
        else:
            S = pyfits.getdata(cdir+'v2_ind'+str(i)+'.fits')
        bl_pix_arr.append(S)
    if redo==False:
        PSall = pyfits.getdata(cdir+'syn_pspec.fits')
    else:
        pyfits.writeto(cdir+'syn_pspec.fits',np.array(PSall),overwrite=True)

    plt.imshow(PSall**0.1,origin='lower')
    for x in range(len(bl_pix_arr)):
        plt.scatter(np.array(bl_pix_arr[x])[:,0],np.array(bl_pix_arr[x])[:,1])
    plt.show()
    
    plt.imshow(Fall,origin='lower')
    plt.show()


    bls_uv = np.array(bls_uv)
    bls_pix = np.array(bls_pix)

    
    ## generating CP sampling coordinates from those calculated for the baselines
    cp_pix_arr = []
    cp_uvs = []
    cps_pix = []
    for i in range(len(mat)):
        mrow = np.array(mat)[i]
        inds = np.where(mrow!=0)[0]
        count=0
        uvtmp=[]
        uvptmp = []
        for ind in inds:
            diff1 = np.sum((bl_pix_arr[ind] - bls_pix[ind])**2,axis=-1)
            diff2 = np.sum((bl_pix_arr[ind] - (np.array([npix,npix])-bls_pix[ind]))**2,axis=-1)
            if mrow[ind] == 1:
                tmp = bl_pix_arr[ind][np.where(diff1 < diff2)]
                uvtmp.append(np.squeeze(np.asarray(bls_uv[ind])))
                uvptmp.append(bls_pix[ind])
                plt.scatter(bl_pix_arr[ind][:,0],bl_pix_arr[ind][:,1],c='grey')
                plt.scatter(tmp[:,0],tmp[:,1],c='b')
                plt.scatter(bls_pix[ind][0],bls_pix[ind][1],c='k')
                plt.axhline(npix/2)
                plt.axvline(npix/2)
            if mrow[ind] == -1:
                tmp = bl_pix_arr[ind][np.where(diff1 > diff2)]
                uvtmp.append(-(np.squeeze(np.asarray(bls_uv[ind]))))
                uvptmp.append(np.array([npix,npix])-bls_pix[ind])
                bls_toplot = np.array([npix,npix])-bls_pix[ind]
                plt.scatter(bl_pix_arr[ind][:,0],bl_pix_arr[ind][:,1],c='grey')
                plt.scatter(tmp[:,0],tmp[:,1],c='r')
                plt.scatter(bls_pix[ind][0],bls_pix[ind][1],c='k')
                plt.axhline(npix/2)
                plt.axvline(npix/2)
            tmp = np.array(tmp)
            pyfits.writeto(cdir+'ind'+str(i)+'_vert'+str(count)+'.fits',tmp,overwrite=True)
            count+=1
        plt.show()
        cp_uvs.append(uvtmp)
        cps_pix.append(uvptmp)
    pyfits.writeto(cdir+'cp_uvs.fits',np.array(cp_uvs),overwrite=True)
    pyfits.writeto(cdir+'cp_pix.fits',np.array(cps_pix),overwrite=True)
    
    cvis_uvs = []
    cvis_pix = []
    for ind in range(len(bls_uv)):
        diff1 = np.sum((bl_pix_arr[ind] - bls_pix[ind])**2,axis=-1)
        diff2 = np.sum((bl_pix_arr[ind] - (np.array([npix,npix])-bls_pix[ind]))**2,axis=-1)
        tmp = bl_pix_arr[ind][np.where(diff1 < diff2)]
        uvtmp = np.squeeze(np.asarray(bls_uv[ind]))
        uvptmp = bls_pix[ind]
        cvis_uvs.append(uvtmp)
        cvis_pix.append(uvptmp)
        pyfits.writeto(cdir+'cvis_ind'+str(ind)+'.fits',np.array(tmp),overwrite=True)
    pyfits.writeto(cdir+'cvis_uvs.fits',np.array(cvis_uvs),overwrite=True)
    pyfits.writeto(cdir+'cvis_pix.fits',np.array(cvis_pix),overwrite=True)
    return

