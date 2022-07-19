# SAMpy
A Fourier-Plane Pipeline for NIRISS AMI Data (and more!)

See Sallum & Eisner 2017 for a description of the pipeline that has morphed into SAMpy, and some 2022 SPIE proceedings (once they exist). This came together thanks to several people (including, but not limited to the following alphabetical list):

- Josh Eisner
- Kenzie Lach
- Shrishmoy Ray
- Steph Sallum
- Christina Vides

This version has just one example notebook for the JWST/NIRISS mask, but you can use the code for arbitrary mask designs. We'll be uploading examples for some ground based setups (e.g. VLT/SPHERE, Keck/NIRC2) soon!

To get started, clone the repository:

```
git clone https://github.com/JWST-ERS1386-AMI/SAMpy
```

Then use conda to create an environment for SAMpy using the SAMpy.yml file:

```
conda env create --file SAMpy
```

Activate the SAMpy environment, and open jupyter notebook:

```
conda activate SAMpy
jupyter notebook
```

Then go ahead and open up the notebook SAMpy_ABDor_CPs+V2s_example.ipynb to see how things work. 


