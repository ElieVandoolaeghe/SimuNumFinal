import numpy as np

def MaxwellBoltzmann_2d_Naive( k_BT = 1., particleMass = 1. ):
    '''
    Returns speed distributed according to
    two-dimensional Maxwell-Boltzmann distribution.
    See for example page 103 and following in
    http://blancopeck.net/Statistics.pdf
    '''
    dim = 2

    vel_mean = 0.0
    standDev = np.sqrt( k_BT / particleMass )

    vel = np.random.normal( vel_mean, standDev, dim )
    #vel = np.array([v_x, v_y])

    speed = np.linalg.norm( vel )
    #speed = sqrt( v_x**2 + v_y**2 )

    return speed

def MaxwellBoltzmann_2d_SemiPro( k_BT = 1., particleMass = 1. ):
    '''
    Returns speed distributed according to
    two-dimensional Maxwell-Boltzmann distribution.
    Gaussian random numbers are sampled following
    http://blancopeck.net/Statistics.pdf
    see page 38 (and 37 for understanding).
    See also page 103 for Maxwell distribution.
    '''
    angle = np.random.uniform(2.*np.pi)
    #random angle

    standDev = np.sqrt( k_BT / particleMass )
    Ypsilon = -np.log( np.random.uniform() )
    speed = standDev * np.sqrt( 2.*Ypsilon )

    v_x = speed * np.cos( angle )
    v_y = speed * np.sin( angle )
    #v_x and v_y are two independent Gaussian random numbers of zero mean and strandard deviation standDev defined above.

    return np.sqrt( v_x**2 + v_y**2 )

def MaxwellBoltzmann_2d_Pro( k_BT = 1., particleMass = 1. ):
    '''
    Returns speed distributed according to
    two-dimensional Maxwell-Boltzmann distribution.
    Speed is sampled following
    http://blancopeck.net/Statistics.pdf
    see page 38 (and 37 for understanding).
    See also page 103 for Maxwell distribution.
    '''
    standDev = np.sqrt( k_BT / particleMass )
    Ypsilon = -np.log( np.random.uniform() )
    speed = standDev * np.sqrt( 2.*Ypsilon )
    return speed


def List_of_speed(x):
    h = []
    for i in range(len(x[:,1])):
        h.append(np.linalg.norm(x[i,:]))
    return h

import matplotlib.pyplot as plt
plt.hist(List_of_speed(mols.vel),bins=40)
plt.title('La distribution des vitesses pour 1000 particules',fontsize = 10)
plt.xlabel('vitesse des particules')
plt.ylabel('nombre de particules')
plt.show()




