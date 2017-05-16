import os
import fnmatch
import math
import numpy as np

def lab(Image):
    # Convert to CIE L*a*b* (CIELAB)
    WhitePoint = np.array([0.950456,1,1.088754])
    Image = xyz(Image)  # Convert to XYZ
    # Convert XYZ to CIE L*a*b*
    X = Image[:,:,0]/WhitePoint[0]
    Y = Image[:,:,1]/WhitePoint[1]
    Z = Image[:,:,2]/WhitePoint[2]
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)
    Image[:,:,0] = 116*fY - 16    # L*
    Image[:,:,1] = 500*(fX - fY)  # a*
    Image[:,:,2] = 200*(fY - fZ)  # b*
    return Image
    
def xyz(Image):
    # Convert to CIE XYZ 
    WhitePoint = np.array([0.950456,1,1.088754])
    # Undo gamma correction
    R = invgammacorrection(Image[:,:,0])
    G = invgammacorrection(Image[:,:,1])
    B = invgammacorrection(Image[:,:,2])
    # Convert RGB to XYZ
    T = np.linalg.inv(np.array([[3.240479,-1.53715,-0.498535],[-0.969256,1.875992,0.041556],[0.055648,-0.204043,1.057311]]))
    Image[:,:,0] = T[0,0]*R + T[0,1]*G + T[0,2]*B  # X 
    Image[:,:,1] = T[1,0]*R + T[1,1]*G + T[1,2]*B;  # Y
    Image[:,:,2] = T[2,0]*R + T[2,1]*G + T[2,2]*B;  # Z
    return Image

def invgammacorrection(Rp):
    R = np.real(((Rp + 0.099)/1.099)**(1/0.45))
    i = R < 0.018
    R[i] = Rp[i]/4.5138
    return R

def f(Y):
    fY = np.real(Y**(1.0/3))
    i = (Y < 0.008856)
    fY[i] = Y[i]*(841.0/108) + (4.0/29)
    return fY

# extensions can be a single tring like '.png' or '.jpg'
# or a list of extensions. they should all be lowercase
# but the . is important.
def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ( len(ext) and ext.lower() in extensions ):
                yield joined

def find_rectangle(n, max_ratio=2):
    sides = []
    square = int(math.sqrt(n))
    for w in range(square, max_ratio * square):
        h = n / w
        used = w * h
        leftover = n - used
        sides.append((leftover, (w, h)))
    return sorted(sides)[0][1]

def make_mosaic(images, nx, ny, channels=3):
    side = int(np.sqrt(len(images[0]) / channels))
    images = images.reshape(-1, side, side, channels)
    image_gen = iter(images)
    mosaic = np.empty((side*ny, side*nx, channels))
    for i in range(ny):
        ia = (i)*side
        ib = (i+1)*side
        for j in range(nx):
            ja = j*side
            jb = (j+1)*side
            mosaic[ia:ib, ja:jb] = next(image_gen)
    return mosaic