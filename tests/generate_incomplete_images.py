import numpy as np

def generate_mask(imgs, xrange, yrange, zrange):
    '''
    Generates a random mask for a 3D image for the model to infill (do it in the corner) 
    
    Inputs:
    imgs (numpy ndarray) - images to generate masks for
    xrange ([int, int]) - x range for which to remove pixels 
    yrange ([int, int]) - y range for which to remove pixels 
    zrange ([int, int]) - z range for which to remove pixels 
    
    Outputs:
    mask (numpy ndarray) - binary mask where 0's indicate where pixels have been removed 
    newimg (numpy ndarray) - masked image 
    '''
    
    numpy_imgs = imgs[:] # in case it's a zarr 
    mask = np.ones(numpy_imgs.shape, dtype='uint8')
    mask[:,xrange[0]:xrange[1], yrange[0]:yrange[1], zrange[0]:zrange[1]] = 0 
    
    newimg = mask*numpy_imgs
    return mask, newimg 

def main():
    ''' The purpose of this module is to randomly generate missing pixels from image patches 
    so that we can test models.'''
    pass
    
if __name__ == "__main__":
    main()