# function to convert RGB value to Lab
import numpy
import pdb
def lab(Image):
    # Convert to CIE L*a*b* (CIELAB)
    WhitePoint = numpy.array([0.950456,1,1.088754])
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
    WhitePoint = numpy.array([0.950456,1,1.088754])
    # Undo gamma correction
    R = invgammacorrection(Image[:,:,0])
    G = invgammacorrection(Image[:,:,1])
    B = invgammacorrection(Image[:,:,2])
    # Convert RGB to XYZ
    T = numpy.linalg.inv(numpy.array([[3.240479,-1.53715,-0.498535],[-0.969256,1.875992,0.041556],[0.055648,-0.204043,1.057311]]))
    Image[:,:,0] = T[0,0]*R + T[0,1]*G + T[0,2]*B  # X 
    Image[:,:,1] = T[1,0]*R + T[1,1]*G + T[1,2]*B;  # Y
    Image[:,:,2] = T[2,0]*R + T[2,1]*G + T[2,2]*B;  # Z
    return Image

def invgammacorrection(Rp):
    R = numpy.real(((Rp + 0.099)/1.099)**(1/0.45))
    i = R < 0.018
    R[i] = Rp[i]/4.5138
    return R

def f(Y):
    fY = numpy.real(Y**(1.0/3))
    i = (Y < 0.008856)
    fY[i] = Y[i]*(841.0/108) + (4.0/29)
    return fY