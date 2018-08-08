import ligate.IO as io
import numpy as np
from ligate.utils import * 
# from ligate.models import * 
import time, zarr

if __name__ == '__main__':
    processData = True # if we need to process the data or not. 
    makeTrainTestData = True # if we still need to make the training and test data after processing 
    save_dir = r'data\lectin594_small_fullres\X.zarr'
    DataFileRange = {'x':all,'y':all,'z':[1,100]} # in case we want to cut out parts of the image. 
    
    if processData:
        img_dir = r'W:\Webster\for_imaris_viewing\top.tif'
        img = io.readData(img_dir, **DataFileRange)
        PreprocessParameter = { 'img': img,
                                'desired_shape': (32,32,32),
                                'overlap': (8,8,8),
                                # 'save_dir': save_dir
                                'save_dir': None # if None, then we load everything on memory                                 
                                }
                                
        start = time.time()
        img_chunks = make_image_chunks_3d(**PreprocessParameter)
        print("Time elapsed for making data:",time.time()-start) 
    
        # temporary: save numpy array because it's much faster 
        if PreprocessParameter['save_dir'] is None:
            zarr.save(save_dir,img_chunks) 
            
            
    ## Next, we load the train and test data 
    if makeTrainTestData:
        X = zarr.open(save_dir,mode='r')
        test_size = 0.2 
        data_dir = r'data\lectin594_small' 
        manual_shuffle=False # Pretty much always keep it like this, it's so slow (but may be necessary if we have really large data)
        filter_dark_threshold=0.25  # the fraction of pixels over which are 0 we don't add them to training set.
        
        start = time.time()
        load_train_val_data(X, test_size, data_dir, manual_shuffle, filter_dark_threshold)
        print("Time elapsed for making train/test data:",time.time()-start)
    
    
    
    