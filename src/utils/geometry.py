from cmath import cos
import pandas as pd
import numpy as np


def cart2sph(xyz:np.array) -> np.array:
    """
    convert Cartesian to spherical, theta is polar angle (from +z), phi is from +x to +y
    Args:
        np.array: [x,y,z]
    Returns:
        np.array: [r, theta, phi],  0<= theta <=pi, 0<= phi <= 2pi
    """
    r = np.sqrt((xyz**2).sum(1))
    theta = np.arccos(xyz[:,2])
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    phi[phi < 0] = phi[phi < 0] + 2*np.pi

    return(np.stack((r,theta,phi), axis=1))


def sph2cart(rtp:np.array) -> np.array:
    """ convert spherical to Cartesian
    Args:
        np.array: [r, theta, phi],  0<= r, 0<= theta <=pi, 0<= phi <= 2pi
    Returns:
        np.array: [x, y, z]
    """
    x = rtp[:,0] * np.sin(rtp[:,1]) * np.cos(rtp[:,2])
    y = rtp[:,0] * np.sin(rtp[:,1]) * np.sin(rtp[:,2])
    z = rtp[:,0] * np.cos(rtp[:,1])
    
    return(np.stack((x, y, z), axis=1))


def sph2Mollweide(thetaphi: np.array) -> np.array:
    """ spherical (viewed from outside) to Mollweide,
         cf. https://mathworld.wolfram.com/MollweideProjection.html
    Args:
        np.array: [theta, phi] in spherical, omit radius ( = 1)
    Returns:
        np.array:[x, y] in Mollweide projection
    """
    azim = thetaphi[:,1]
    azim[azim > np.pi] = azim[azim > np.pi] - 2*np.pi #longitude/azimuth
    elev = np.pi/2 - thetaphi[:,0] #lattitude/elevation in radian

    N = len(azim) #number of points
    xy = np.zeros((N,2)) #output
    for i in range(N):
        theta = np.arcsin(2*elev[i]/np.pi)
        if np.abs(np.abs(theta) - np.pi/2) < 0.001:
            xy[i,] = [2*np.sqrt(2)/np.pi*azim[i]*np.cos(theta), np.sqrt(2)*np.sin(theta)]
        else:
            # to calculate theta 
            dtheta = 1 
            while dtheta > 1e-3:
                theta_new = theta -(2*theta +np.sin(2*theta) -np.pi*np.sin(elev[i]))/(2+2*np.cos(2*theta))
                dtheta = np.abs(theta_new - theta)
                theta = theta_new
            xy[i,] = [2*np.sqrt(2)/np.pi*azim[i]*np.cos(theta), np.sqrt(2)*np.sin(theta)]
    return xy


def sph2Mercator(thetaphi: np.array) -> np.array:
    """ spherical (viewed from outside) to Mercator
        cf. https://mathworld.wolfram.com/MercatorProjection.html
    Args:
        np.array: [theta, phi] in spherical, omit radius ( = 1)
    Returns:
        np.array:[x, y] in Mercator projection
    """
    azim = thetaphi[:,1]
    azim[azim > np.pi] = azim[azim > np.pi] - 2*np.pi #longitude/azimuth
    elev = np.pi/2 - thetaphi[:,0] #lattitude/elevation in radian

    xy = np.stack((azim, np.log(np.tan(np.pi/4 + elev/2))), axis=1)
    return xy

def plane_square(
        vn:np.array
        , vt:np.array
        , mf = 2
        ) -> np.array:
    """ 
    for a given normal vector, construct a 3d rotation matrix as a product of 
    first a rotaion around y-axis, then a second around z-axis

    Parameters
    ----------
    vn : np.array
        normal vector of the plane
    vt : np.array
        translation vector of the plane from the origin
    mf : float
        magnification factor of the plane

    Returns
    -------
    pl_rot : np.array
        a 4x3 matrix of the vetices of the desired square
    """
    
    # initialize a unit plane
    pl = pd.DataFrame(columns=['x', 'y', 'z'])
    pl['x'] = [0.5, 0.5, -.5, -.5]
    pl['y'] = [.5, -.5, .5, -.5]
    pl['z'] = [0,0,0,0]
    pl = pl * mf # scale the plane

    # compute rotation matrices
    vxy = np.array([vn[0], vn[1], 0]) # projection on xy plane
    theta = np.arccos(vn @ np.array([0,0,1]) / (np.linalg.norm(vn) * 1)) # angle between vn and z-axis
    phi = np.arccos(vxy @ np.array([1,0,0]) / (np.linalg.norm(vxy) * 1)) + np.pi # angle between vxy and x-axis
    Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) # rotation around y-axis
    Rz = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]]) # rotation around z-axis

    pl_rot = pl @ Ry @ Rz # rotate the plane
    pl_rot.columns = ['x', 'y', 'z'] # change columna names

    # translate the plane to a given point
    pl_rot = pl_rot + vt
    
    return(pl_rot)