from kernelized_sorting_color import KS
from utils import *
import matplotlib.pylab as mp
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Kernelized Sorting for Image Layouting with color features.')
parser.add_argument('-s', '--image_size',
    type=int,
    default=40,
    help='Size of the side each tile.')
parser.add_argument('-i', '--input_directory',
    type=str,
    default='images',
    help='Directory containing images.')
parser.add_argument('-o', '--output_filename',
    type=str,
    default='output.png',
    help='Output filename.')
parser.add_argument('-n',
    type=int,
    default=None,
    help='Number of tiles along x axis.')
args = parser.parse_args()

nx = args.n
psize = args.image_size
iname = args.input_directory
oname = args.output_filename

filenames = list(list_all_files(iname, ['.png', '.jpg']))
n = len(filenames)
if nx is None:
    nx, ny = find_rectangle(n)
else:
    ny = n / nx
print('Using {}x{} tiles.'.format(nx, ny))

print('Resizing images...')
imgdata = []
data = []
for fn in filenames:
    im = Image.open(fn)
    w, h = im.size
    hw, hh = (w/2, h/2)
    hs = min(hw, hh)
    box = (hw - hs, hh - hs, hw + hs, hh + hs)
    im = im.crop(box)
    im = im.resize((psize, psize), Image.LANCZOS)

    aim = np.asarray(im)
    data.append(aim.flatten())

    # convert from RGB to Lab
    daim = np.double(aim)/255.0
    daimlab = lab(daim)
    imgdata.append(daimlab.flatten())
        
data = np.asarray(data)
imgdata = np.asarray(imgdata)
griddata = np.zeros((2,ny*nx))
griddata[0,] = np.kron(range(1,ny+1),np.ones((1,nx)))
griddata[1,] = np.tile(range(1,nx+1),(1,ny))

# do kernelized sorting procedure
PI = KS(imgdata, griddata.T)

# build and show image
sorted_indices = PI.argmax(axis=1)
mosaic = make_mosaic(data[sorted_indices], nx, ny)
im = Image.fromarray(mosaic.astype(np.uint8))
im.save(oname)
