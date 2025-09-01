# Loading the packages and dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def create_particle(box_size=9, resolution=9, mu=0.0, sigma=0.5):
    """
    Parameters
    ----------
    box_size : int
        Size of the box in which the particle is created.
    resolution : int
        Resolution of the box in which the particle is created.
    mu : float
        Mean of the Gaussian-like distribution.
    sigma : float
        Standard deviation of the Gaussian-like distribution.
    Returns
    -------
    g : 2D ndarray
        2D Gaussian-like array.
    """
    
    half_box = int((box_size-1)/2)
    
    x, y = np.meshgrid(np.linspace(-half_box,half_box,resolution), np.linspace(-half_box,half_box,resolution))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    print('2D Gaussian-like array created')
    # print(g)
    # g = g-g.min()
    # g = g/g.max()
    return g


def gaussian_noisy(image, mean, sigma):
    """
    Parameters
        image : ndarray
            Input image data. Will be converted to float.
        mean : float
           The average of the Gaussian distribution used to sample noise
        var: float
           The variance of the Gaussian distribution used to sample noise
        npseed : int
            Seed for the random number generator for reproducibility.
    Returns
        noisy : ndarray
            Noisy image data.  
    """

    row,col= image.shape
    # mean = 2
    # var = 0.05
#     sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


def create_noisy_particle(n_frames, noise_sigma, array_size, save_path='none', npseed=1):
    """
    Parameters
    ----------
    n_frames : int
        Number of frames to be created.
    noise_sigma : float
        Standard deviation of the Gaussian noise to be added.
    array_size : int
        Size of the square array (array_size x array_size).
    save_path : str
        Path to save the generated images.
    npseed : int
        Seed for the random number generator for reproducibility.
    Returns
    -------
    images_dict : dict
        Dictionary containing the generated noisy images.
    """
    
    particle_image = create_particle(box_size=7, resolution=7, mu=0.0, sigma=6.7/4)
    
    position = int(array_size/2)
    particle_size = particle_image.shape[0]
    
    noise_array = np.meshgrid(np.linspace(0,0,array_size), np.linspace(0,0,array_size))[0]
    noise_array[(position-int(particle_size/2)):(position+(int(particle_size/2)+1)),
                (position-int(particle_size/2)):(position+(int(particle_size/2)+1))] = particle_image
    
    images_dict = {}
    np.random.seed(seed=npseed)
    for fr in range(n_frames):
#         print(fr)
        noisy_image = gaussian_noisy(noise_array, 0, noise_sigma)
        images_dict[fr] = noisy_image+noise_array  # scale the image above 0 by subtracting the negative mean
    
        plt.imshow(images_dict[fr], cmap='gray')
        plt.colorbar()
        plt.yticks([])
        plt.xticks([])
        plt.text(2,-2, 'frame: '+str(fr), fontsize=12, fontweight='bold', color='black')
        if os.path.isdir(save_path):
            plt.savefig(save_path+'/'+str(fr)+'noisy_particle.jpeg')
        plt.close()
    
    return images_dict