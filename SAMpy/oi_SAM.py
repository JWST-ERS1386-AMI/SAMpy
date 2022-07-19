from .oifits import *
import math


def rotate(point,deg):
    s, c = [f(np.radians(deg)) for f in (math.sin, math.cos)]
    dx, dy = (c*point[0] - s*point[1], s*point[0] + c*point[1])
    return (dx,dy)

def rot_cp_uvs(PAs,cdir):
    uvs = []
    cp_uvs = pyfits.getdata(cdir+'cp_uvs.fits')
    for r in PAs:
        uvs.append([[rotate(cp_uvs[y][j],-r) for j in range(len(cp_uvs[0]))] for y in range(len(cp_uvs))])
    return np.array(uvs)

def rot_bl_uvs(PAs,cdir):
    uvs = []
    bl_uvs = pyfits.getdata(cdir+'bl_uvs.fits')
    for r in PAs:
        uvs.append([rotate(bl_uvs[y], -r) for y in range(len(bl_uvs))])
    return np.array(uvs)


def build_oi(targ,lam,cps,cerrs,v2s,verrs,PAs,cdir,fn):
    oi_arr = oifits()
    oi_arr.wavelength['ARRAY'] = OI_WAVELENGTH(lam*1.0e-06)
    oi_arr.target = np.append(oi_arr.target,OI_TARGET(targ,0,0))
    v2_uvs = rot_bl_uvs(PAs,cdir)
    cp_uvs = rot_cp_uvs(PAs,cdir)
    for x in range(len(v2s)):
        for i in range(len(v2s[x])):
            oi_arr.vis2 = np.append(oi_arr.vis2,OI_VIS2(
                timeobs = datetime.datetime(2000,1,1,0,0,0),
                int_time = 0.0,
                vis2data = v2s[x,i],
                vis2err = verrs[x,i],
                flag = np.array([False],dtype=bool),
                ucoord = v2_uvs[x,i,0],
                vcoord = v2_uvs[x,i,1],
                wavelength = oi_arr.wavelength['ARRAY'],
                target = oi_arr.target[0]))
        for i in range(len(cps[x])):
            oi_arr.t3 = np.append(oi_arr.t3,OI_T3(
                timeobs = datetime.datetime(2000,1,1,0,0,0),
                int_time = 0.0,
                t3amp = 0,
                t3amperr = 0,
                t3phi = cps[x,i],
                t3phierr = cerrs[x,i],
                flag = np.array([False],dtype=bool),
                u1coord = cp_uvs[x,i,0,0],
                v1coord = cp_uvs[x,i,0,1],
                u2coord = cp_uvs[x,i,1,0],
                v2coord = cp_uvs[x,i,1,1],
                wavelength = oi_arr.wavelength['ARRAY'],
                target = oi_arr.target[0]))
    oi_arr.save(fn)
    return