from scipy.signal import medfilt2d as medfilt
import emcee
from scipy import optimize,ndimage
import numpy as np

def find_psf_center(img, verbose=True, nbit=10):
    ''' Name of function self explanatory: locate the center of a PSF.

    ------------------------------------------------------------------
    Uses an iterative method with a window of shrinking size to
    minimize possible biases (non-uniform background, hot pixels, etc)

    Options:
    - nbit: number of iterations (default 10 is good for 512x512 imgs)
    - verbose: in case you are interested in the convergence
    ------------------------------------------------------------------ '''
    temp = img.copy()
    bckg = np.median(temp)   # background level
    temp -= bckg
    mfilt = medfilt(temp, 3) # median filtered, kernel size = 3
    (sy, sx) = mfilt.shape   # size of "image"
    xc, yc = sx/2, sy/2      # first estimate for psf center

    signal = np.zeros_like(img)
    signal[mfilt > 0.1*mfilt.max()] = 1.0

    #for it in xrange(nbit):
    for it in range(nbit):
        sz = sx/2/(1.0+(0.1*sx/2*it/(4*nbit)))
        x0 = np.max([int(0.5 + xc - sz), 0])
        y0 = np.max([int(0.5 + yc - sz), 0])
        x1 = np.min([int(0.5 + xc + sz), sx])
        y1 = np.min([int(0.5 + yc + sz), sy])

        mask = np.zeros_like(img)
        mask[y0:y1, x0:x1] = 1.0

        profx = (mfilt*mask*signal).sum(axis=0)
        profy = (mfilt*mask*signal).sum(axis=1)

        xc = (profx*np.arange(sx)).sum() / profx.sum()
        yc = (profy*np.arange(sy)).sum() / profy.sum()

        #pdb.set_trace()

        if verbose:
            print("it #%2d center = (%.2f, %.2f)" % (it+1, xc, yc))

    return (xc, yc)

# =========================================================================
# =========================================================================

def gauss_smooth_im(inim,lam,d,psc):
    sig = (lam*1.0e-6)/d*206265.0/psc/2.35482
    outim = ndimage.filters.gaussian_filter(inim,sigma = sig)
    return outim

def get_center(im,lam,d,psc):
    im2 = gauss_smooth_im(im,lam,d,psc)
    #im2 = ndimage.filters.uniform_filter(im,20)
    pos = np.where(im2==np.max(im2))
    y,x = pos[0][0],pos[1][0]
    return y,x


def fourier_center(im,dyn,dxn):
    FTim = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im)))
    usbig = np.fft.fftshift(np.fft.fftfreq(len(FTim)))
    FToff = np.outer(np.exp(-2.0*np.pi*(usbig*dyn)*1.0j),np.exp(-2.0*np.pi*usbig*dxn*1.0j))
    return FTim,FTim*FToff


#yint,xint = get_center(imt,lamc,10.0,0.009)
#x,y = find_psf_center(imt)
#dy,dx = y-yint,x-xint
#fft_cen = fourier_center(imt,dy,dx)
#FTt = fft_cen/(np.max(np.abs(fft_cen))*float(len(locs)))