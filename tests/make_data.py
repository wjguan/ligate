import ligate.IO as io
import numpy as np
from ligate.utils import * 
# from ligate.models import * 
import time, zarr

if __name__ == '__main__':
    processData = True # if we need to process the data or not. 
    makeTrainTestData = True # if we still need to make the training and test data after processing 
    save_dir = r'data\lectin594_large\X_all.zarr'
    
    if processData:
        img_dir = r'data\lectin594\top_resampled_4x.tif'
        img = io.readData(img_dir)
        PreprocessParameter = { 'img': img,
                                'desired_shape': (100,100,50),
                                'overlap': (16,16,16),
                                # 'save_dir': save_dir
                                'save_dir': None
                                }
                                
        start = time.time()
        img_chunks = make_image_chunks_3d(**PreprocessParameter)
        print(time.time()-start) 
    
        # temporary: save numpy array because it's much faster 
        if PreprocessParameter['save_dir'] is None:
            zarr.save(save_dir,img_chunks) 
            
            
    ## Next, we load the train and test data 
    if makeTrainTestData:
        X = zarr.open(save_dir,mode='r')
        test_size = 0.2 
        data_dir = r'data\lectin594_large' 
        manual_shuffle=False
        
        start = time.time()
        load_train_val_data(X, test_size, data_dir, manual_shuffle)
        print(time.time()-start)
    
    
    
    