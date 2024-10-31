from scipy.constants import physical_constants, c
import pmd_beamphysics
import cheetah
import torch
import numpy as np
import scipy

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


def ParameterBeam_to_ParticleGroup(n,ParameterBeam):
    """
    This function converts a Cheetah ParameterBeam into a OpenPMD ParticleGroup.  A ParameterBeam defines the first- and second-order moments of the beam distribution function, so some assumptions need to be made in order to make a distribution of particles.  Namely, this function assumes that the distribution is gaussian and takes quasi-random samples from a Gaussian distribution to make a beam with first- and second-order moments that satisfy the requirements in the ParameterBeam.  

    Argument: 
    n: int, number of particles in ParticleGroup distribution
    ParameterBeam: Cheetah ParameterBeam object that defines the first- and second-order mements of the beam distribution function 

    Returns: OpenPMD ParticleGroup object of the beam distribution
    """
    dist = ParameterBeam_to_Dist(n,ParameterBeam)
    particle_group = dist_to_ParticleGroup(dist)
    return particle_group

def ParticleBeam_to_ParticleGroup(cheetah_beam):
    """
    Converts an Cheetah ParticleBeam into an openPMD ParticleGroup format.
    :param cheetah_beam: The input beam object containing x, y, px, py, tau, p, and energy.
    :return: ParticleGroup containing the beam information.
    """
    dist = ParticleBeam_to_Dist(cheetah_beam)

    particle_group = dist_to_ParticleGroup(dist)
    return particle_group

def particles_to_dist(particles,energy,charge):
    """
    This function takes a distribution of particles and reformats it into a dictionary

    Arguments:
    particles -- matrix of particles
    charge -- float of charge
    energy -- float of energy
    """
    if isinstance(particles,np.ndarray):
        dist_mat = particles
    else: 
        dist_mat = particles.numpy()
    try: 
        charge = float(charge)
    except:
        raise ValueError("Charge is not a float")
        
    try: 
        energy = float(energy)
    except:
        raise ValueError("Energy is not a float")
        
    # for i in range(np.shape(dist_mat)[0]):
    #     dist_mat[i,:] = dist_mat[i,:]
    dist = {'x':dist_mat[0,:],'y':dist_mat[2,:],'px':dist_mat[1,:],'py':dist_mat[3,:],
             'z':dist_mat[4,:],'pz':dist_mat[5,:],'energy':energy,'charge':charge}
    return dist

def ParameterBeam_to_Dist(n,ParameterBeam):
    """
    This function converts a Cheetah ParameterBeam into a distribution of positions and momenta.  These are returned in a dictionary.  See notes above for how this is accomplished

    Argument: 
    n: int, number of particles in ParticleGroup distribution
    ParameterBeam: Cheetah ParameterBeam object that defines the first- and second-order mements of the beam distribution function 

    Returns: dictionary specifying the beam distribution
    """
    # Get the covariance matrix and make a distribution
    cov_mat = np.squeeze(ParameterBeam._cov.numpy())[:6,:6]
    dist_mat = Gaussian_Dist_From_Cov_Mat(n,cov_mat) 

    mu_mat = np.squeeze(ParameterBeam._mu.numpy())

    # Add back the mean in each dimension
    for i in range(np.shape(dist_mat)[0]):
        dist_mat[i,:] = dist_mat[i,:] + mu_mat[i]
    # Make the dictionary
    dist = particles_to_dist(dist_mat,ParameterBeam.energy,ParameterBeam.total_charge)
    return dist

def ParticleBeam_to_Dist(cheetah_beam):
    """
    This function converts a Cheetah ParticleBeam into a distribution of positions and momenta.  These are returned in a dictionary.

    Argument: 
    cheetah_beam: Cheetah ParticleBeam object that defines the beam distribution

    Returns: dictionary specifying the beam distribution
    """
    try:
        x = cheetah_beam.x.numpy().reshape(-1)
        y = cheetah_beam.y.numpy().reshape(-1)
        px = cheetah_beam.px.numpy().reshape(-1)
        py = cheetah_beam.py.numpy().reshape(-1)
        z = cheetah_beam.tau.numpy().reshape(-1)
        pz = cheetah_beam.p.numpy().reshape(-1)
        energy = cheetah_beam.energy.numpy()
        charge = cheetah_beam.total_charge.numpy()
    
        dist = {'x':x,'y':y,'px':px,'py':py,'z':z,'pz':pz,'energy':energy,'charge':charge}
    except:
        dist = particles_to_dist(torch.transpose(cheetah_beam.particles,0,1),cheetah_beam.energy,cheetah_beam.total_charge)
    return dist


def dist_to_ParticleGroup(dist):
    """
    This function converts a distribution of positions and momenta specified in a dictionary into an OpenPMD ParticleGroup object.  
    
    Argument: 
    dist: dictionary specifying the beam distribution

    Returns: ParticleGroup containing the beam information.
    """
    
    
    p_ref = np.sqrt(dist['energy']**2 - electron_mass_eV**2)

    gamma = np.sqrt(1 + (dist['pz'] * p_ref / electron_mass_eV)**2)
    beta_z = np.sqrt(1 - 1 / gamma**2)
    # c = 3e8
    data = {
        'x': dist['x'],
        'y': dist['y'],
        'z': dist['z'],
        'px': dist['px'] * p_ref,
        'py': dist['py'] * p_ref,
        'pz': (1 + dist['pz']) * p_ref,
        't': dist['z']/(beta_z*c),
        'weight': np.ones_like(dist['x']).reshape(-1) / dist['x'].shape[0] * dist['charge'],
        'status': np.ones_like(dist['x']).reshape(-1),
        'species': 'electron'
    }
    particle_group = pmd_beamphysics.ParticleGroup(data=data)
    return particle_group

def Gaussian_Dist_Maker(n,mu,sigma,lSig,rSig):
    """
    This function returns a truncated gaussian distribution of quasi-random particles.  This uses the Halton series
    
    Argument:
    n -- int number of particles
    mu -- float: center of distribution/mean
    sigma -- float: std of distribution
    lSig -- float number of sigma at which to truncate Gaussian left
    rSig -- float number of sigma at which to truncate Gaussian right

    Returns: 1-D Gaussian dist of quasi-random numbers
    """
    # Check inputs
    try: n = int(n)
    except: raise ValueError("n is not an int!")
    
    try: mu = float(mu)
    except: raise ValueError("mu is not a float!")
    
    try: sigma = float(sigma)
    except: raise ValueError("sigma is not a float!")
    
    try: lSig = float(lSig)
    except: raise ValueError("lSig is not a float!")
    
    try: rSig = float(rSig)
    except: raise ValueError("rSig is not a float!")
    
    
    # get and shuffle n samples from halton series
    h=scipy.stats.qmc.Halton(1)
    X0=h.random(n=n)
    np.random.shuffle(X0)
    
    # Make these into Gaussian and return
    X0=X0*(1-(1-scipy.stats.norm.cdf(lSig))-(1-scipy.stats.norm.cdf(rSig)))
    X0=X0+(1-scipy.stats.norm.cdf(lSig))
    GaussDist = mu + np.sqrt(2)*sigma*scipy.special.erfinv(2*X0-1)
    return np.squeeze(GaussDist)

def Gaussian_Dist_From_Cov_Mat(n,cov_mat):
    """
    This function returns a n correlated samples of dimension of the covariance matrix (cov_mat).  
    These samples are a gaussian distribution that satisfies this covariance matrix.
    This makes use of Cholesky decomposition.
    
    Argument:
    n -- int, number of particles
    cov_mat --  2-D numpy array representing the covariance matrix 

    Returns: a distribution according to the covariance matrix
    """
    # Check inputs
    try: n = int(n)
    except: raise ValueError("n is not an int!")
    cov_mat = np_array_dim_checker(cov_mat)
    assert np.shape(cov_mat)[0]==np.shape(cov_mat)[1]
    
    # Make particles with independent Gaussians   
    Z=np.zeros([n,np.shape(cov_mat)[0]])
    for i in range(np.shape(cov_mat)[0]):
        Z[:,i]=Gaussian_Dist_Maker(n,0,1,10,10)
        
    # Use Cholesky distribution to make correlated samples, return
    Q=np.linalg.cholesky(cov_mat)
    x=np.matmul((Q),np.transpose(Z))
    return x

def np_array_dim_checker(img,dim=2):
    """
    This function checks to make sure that the input is indeed a N-D numpy array.  Throws an error if not.  Returns image in a guaranteed numpy array
    
    Argument:
    img -- input to be tested
    dim -- optional int representing the dimensionalty of the desired numpy array.
    
    """
    # Check inputs 
    assert isinstance(dim,int)==True, "Dimension is not an integer!"
    
    # Convert img to np array 
    try:
        img=np.array(img)
    except:
        raise ValueError("img cannot be converted to a numpy array.")
    
    # Check dimensions
    img=np.squeeze(img)
    assert len(np.shape(img))==dim, "The shape of the image provided is not " + str(dim)+"-D"
    return img


