## Kernelized Sorting for Image Layouting with bag-of-words features
## param[in] psize sampling size of image
## param[in] ino width of the layout area
## param[in] jno height of the layout area
## param[in] iname file name for the images
## param[in] bname file name for the bow files

from kernelized_sorting_bow import KS
from utils import lab
import matplotlib.pylab as mp
import Image
import numpy
import pdb

ino = 25
jno = 25
psize = 200
iname = 'PPMImages/'
bname = 'OlympicGames_bows/'

n_image = ino*jno
imgdata = []
data = []
counter = 0
for i in xrange(n_image):
    counter = counter + 1
    fname = iname + str(counter) + '.jpg'
    im = Image.open(fname)
    aim = numpy.asarray(im)
    [M,N,L] = aim.shape
    mno = numpy.fix(M/psize)
    nno = numpy.fix(N/psize)
    aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
    data.append(aim.flatten())          #images
    fname = bname + str(counter) + '.bow'
    lines = open(fname).readlines()
    temp = []
    tokens = lines[0].split()
    for i in xrange(len(tokens)):
        temp.append(eval(tokens[i]))
    temp = numpy.array(temp)   
    imgdata.append(temp.flatten())   #its corresponding bow (bag-of-words features)
    
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
