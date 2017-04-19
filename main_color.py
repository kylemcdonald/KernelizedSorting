## Kernelized Sorting for Image Layouting with color features
## param[in] psize sampling size of image
## param[in] ino width of the layout area
## param[in] jno height of the layout area
## param[in] iname file name for the images

from kernelized_sorting_color import KS
from utils import lab
import matplotlib.pylab as mp
import Image
import numpy
import pdb
ino = 16
jno = 20
psize = 40
iname = 'images/img'

imgdata = []
data = []
counter = 0
for i in xrange(ino):
    for j in xrange(jno):
        counter = counter + 1
        fname = iname + str(counter) + '.jpg'
        im = Image.open(fname)
        aim = numpy.asarray(im)
        [M,N,L] = aim.shape
        mno = numpy.fix(M/psize)
        nno = numpy.fix(N/psize)
        aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
        data.append(aim.flatten())
        daim = numpy.double(aim)/255.0
        # convert from RGB to Lab
        daimlab = lab(daim)
        imgdata.append(daimlab.flatten())
        
data = numpy.array(data)
imgdata = numpy.array(imgdata)
griddata = numpy.zeros((2,ino*jno))
griddata[0,] = numpy.kron(range(1,ino+1),numpy.ones((1,jno)))
griddata[1,] = numpy.tile(range(1,jno+1),(1,ino))

# do kernelized sorting procedure
PI = KS(imgdata,griddata.T)
i_sorting = PI.argmax(axis=1)
imgdata_sorted = data[i_sorting,]
irange = range(0,psize*ino,psize)
jrange = range(0,psize*jno,psize)
patching = numpy.zeros((ino*psize, jno*psize, 3))
for i in xrange(ino):
    for j in xrange(jno):
        patching[irange[i]:irange[i]+psize,jrange[j]:jrange[j]+psize,:] = numpy.reshape(imgdata_sorted[(i)*jno+j,], [psize,psize,3]);

im = Image.fromarray(patching.astype(numpy.uint8))
im.show()
