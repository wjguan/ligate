import argparse, os, zarr, sys, glob
import numpy as np

import ligate.IO as io
from ligate.utils import * 
from distutils.util import strtobool
import matplotlib.pyplot as plt 

from tests.generate_incomplete_images import generate_mask


def predict():
    ## Set GPU using tf 
    import tensorflow as tf 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    from keras.backend.tensorflow_backend import set_session
    set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser(description='Parse model parameters.')
    parser.add_argument("--model_type", help="Gated or vanilla [Required]", type=str, metavar="TYPE")
    parser.add_argument("--checkpoint_file", help="Checkpoint file [Required]", type=str, metavar="FILE_PATH")
    parser.add_argument("--load_data_file", help="zarr file with incomplete images. [REQUIRED]",type=str, metavar="LOAD_FILE")
    parser.add_argument("--batch_size", help="Batch size at prediction (default: 1)",type=int, metavar="INT")
    parser.add_argument("--temperature", help="Temperature value for sampling diverse values (default: 1.0)",type=float, metavar="FLOAT")
    parser.add_argument("--save_dir", help="Root directory where to save generated images (default: results\pixelcnn3d_pred)", type=str, metavar="DIR_PATH")
    parser.add_argument("--normalization_root", help="Root directory containing 'mean.npy' and 'std.npy' for normalization. [REQUIRED]", type=str, metavar="NORM_PATH")
    parser.add_argument("--nb_images", help="Number of images to inpaint. (default: 8)", type=int, metavar="INT") 
    parser.add_argument("--mask_path", help="Path to the file containing all of the masks denoting which parts of the images to predict.", type=str)
    
    # These should be the same parameters as that of the old model.
    ## TODO: automatically populate these based on exported model parameters
    parser.add_argument("--loss", help="Loss function (categorical_crossentropy or discretized_mix_logistic_loss)", metavar="LOSS")
    parser.add_argument("--nb_res_blocks", help="Number of residual blocks. (default: 1)", metavar="INT")
    parser.add_argument("--nb_filters", help="Number of filters for each layer (or double this amount). (default: 128)", metavar="INT")
    parser.add_argument("--filter_size_1st", help="Filter size for the first layer. (default: (7,7,7))", metavar="INT,INT,INT")
    parser.add_argument("--filter_size", help="Filter size for the subsequent layers. (default: (3,3,3))", metavar="INT,INT,INT")
    parser.add_argument("--pad", help="Whether to pad images or not (default: False)", type=str, metavar="BOOL")
    parser.add_argument("--target_size", help="Target shape of the model. (default: (42,42,42))", metavar="INT,INT,INT")
    parser.add_argument("--dropout", help="")
    
    args = parser.parse_args()
    temperature = float(args.temperature) if args.temperature else 1.0 
    batch_size = int(args.batch_size) if args.batch_size else 1 
    
    try:
        checkpoint_file = args.checkpoint_file
        load_data_file = args.load_data_file
        # mask_path = args.mask_path # not yet required 
        norm_root = args.normalization_root 
    except ValueError:
        sys.exit("Error: --checkpoint_file, --normalization_root, and --load_data_file must be specified.")
    
    
    save_dir = args.save_dir if args.save_dir else r'results\pixelcnn3d_pred'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ### Load in data
    X_incomplete = zarr.open(load_data_file, mode='r')
    input_size = X_incomplete.shape[1:]
    
    ### build PixelCNN model ###
    model_params = {}
    model_params['input_size'] = input_size
    if args.nb_res_blocks:
        model_params['nb_res_blocks'] = int(args.nb_res_blocks)
    if args.nb_filters:
        model_params['nb_filters'] = int(args.nb_filters)
    if args.filter_size_1st:
        model_params['filter_size_1st'] = tuple(map(int, args.filter_size_1st.split(',')))
    if args.filter_size:
        model_params['filter_size'] = tuple(map(int, args.filter_size.split(',')))
    if args.pad:
        model_params['pad'] = strtobool(args.pad)
    if args.loss:
        model_params['loss'] = args.loss
    if args.dropout:
        model_params['dropout'] = strtobool(args.dropout)
    
    if args.model_type=='vanilla':
        from ligate.models import PixelCNN3D
        print("it's not gated")
    elif args.model_type=='gated':
        from ligate.bigated3d import BiGatedPixelCNN3D as PixelCNN3D
        print("it's gated")
    pixelcnn = PixelCNN3D(**model_params)
    pixelcnn.build_model()
    pixelcnn.model.load_weights(checkpoint_file) 
    
    
    
    ## Predict one pixel, one image at a time (mask could be different, or we might need to do it in series) 
    nb_images = int(args.nb_images) if args.nb_images else 8 
    target_size = tuple(map(int, args.target_size.split(','))) if args.target_size else (42,42,42)
    
    if not args.mask_path:
        masks, incomplete_imgs = generate_mask(X_incomplete, [32,44],[30,42],[20,32])
    else:
        masks = zarr.open(mask_path, mode='r')
    

    ## Shuffle the indices to change which ones we predict 
    # tobeshuffled = np.arange(X_incomplete.shape[0])
    # np.random.shuffle(tobeshuffled)
    # X_incomplete = X_incomplete[:][tobeshuffled]
    # incomplete_imgs = incomplete_imgs[tobeshuffled] 
    # print(tobeshuffled[0], tobeshuffled[1]) # 311, 157
    # print(np.sum(X_incomplete[0] - incomplete_imgs[0]), np.sum(X_incomplete[1]-incomplete_imgs[1])) # difference between first occluded and actual images 
    # X_pred_same = X_incomplete[:nb_images].copy() # because the predicted and input sizes are different, but we need to feed in the predicted as input
    
    ## This is just for testing: on an image we know has salient information 
    X_incomplete = X_incomplete[311:312]
    for k in range(nb_images//2):
        X_incomplete = np.concatenate((X_incomplete, X_incomplete),axis=0)
    X_pred_same = X_incomplete.copy() 
    
    # Might have to trim the image based on padding s
    if model_params['pad']:
        X_pred = X_incomplete[:nb_images].copy()
    else:
        extrap = [(input_size[i]-target_size[i])//2 for i in range(3)]
        X_pred = X_incomplete[:nb_images, extrap[0]:-extrap[0], extrap[1]:-extrap[1], extrap[2]:-extrap[2]].copy()
    
    ## Parallelize computations 
    if model_params['pad']:
        mask_img = masks[0]
    else:
        mask_img = masks[0,extrap[0]:-extrap[0], extrap[1]:-extrap[1], extrap[2]:-extrap[2]]
    inds = np.argwhere(mask_img==0) # sorts in order of rows, columns, z, i.e. it iterates through z first, then columns, then rows 
         
    # Important - we need to make sure we predict in the order of columns, rows, z (otherwise will be conditioning on wrong information) 
    inds = inds[np.lexsort((inds[:,0],inds[:,2]))]
    for i in range(len(inds)):
        # x = image2kerasarray(X_pred_same, norm_root) #  normalize - or not. 
        ## This is if we stack with ones 
        # x = np.stack([X_pred_same, np.ones(X_pred_same.shape)], axis=-1)
        # x_with_mask = np.stack([X_pred_same*masks[:nb_images], np.ones(X_pred_same.shape), masks[:nb_images]], axis=-1)
        
        
        ## THis is if we don't stack with ones 
        x = np.expand_dims(X_pred_same, axis=-1)
        x_with_mask = np.stack([X_pred_same*masks[:nb_images], masks[:nb_images]] ,axis=-1)
        
        x_with_mask = np.rot90(np.flip(x_with_mask, axis=3), k=2, axes=(0,1))
        next_X_pred = pixelcnn.model.predict([x, x_with_mask], batch_size)
        
        sampled_pred = next_X_pred[:,inds[i,0],inds[i,1],inds[i,2],:]
        
        ## short experiment to see what kinds of distributions ar elearned 
        # plt.plot(np.arange(256), sampled_pred[0])
        # plt.show()
        print([np.argmax(sampled_pred[l]) for l in range(nb_images)])
        
        
        if args.loss == 'categorical_crossentropy':
            sampled_pred = np.array([sample(sampled_pred[j], temperature=temperature) for j in range(len(sampled_pred))])
            
        else:
            ## Sample the discretized logistic distribution 
            sampled_pred = sample_from_discretized_mix_logistic(sampled_pred, nr_mix=10)
        # print("Predicted intensity:",sampled_pred[0],"Actual intensity:",X_incomplete[0,inds[i,0],inds[i,1],inds[i,2]])
        print("Predicted intensities:",[sampled_pred[l] for l in range(nb_images)],"\nActual intensity:",X_incomplete[0,inds[i,0],inds[i,1],inds[i,2]])    
        X_pred[:,inds[i,0],inds[i,1],inds[i,2]] = sampled_pred # output size image 
        if model_params['pad']:
            X_pred_same[:,inds[i,0],inds[i,1],inds[i,2]] = sampled_pred
        else:
            X_pred_same[:,inds[i,0]+extrap[0],inds[i,1]+extrap[1],inds[i,2]+extrap[2]] = sampled_pred # input size image 
    # Since this is a test, we will save as tiffs 
    X_completed = X_incomplete.copy()
    if not model_params['pad']:
        X_completed[:,19:31,12:24,33:45] = X_pred[:,15:27,8:20,29:41] # very specific to the current set up 
    else:
        X_completed = X_pred
    
    for w in range(len(X_pred)):
        # io.writeData(os.path.join(save_dir, 'test_%d_pred.tif'%w),X_pred[w].astype('uint8'))
        io.writeData(os.path.join(save_dir, 'test_%d_pred.tif'%w),X_completed[w].astype('uint8'))
    io.writeData(os.path.join(save_dir, 'test_%d_actual.tif'%w),X_incomplete[0].astype('uint8'))
    io.writeData(os.path.join(save_dir, 'test_%d_incomp.tif'%w),incomplete_imgs[311].astype('uint8'))
    
    

    ## The following is for actual runs (where we cannot parallelize as easily generation of images)
    # for w in range(nb_images):
        # if model_params['pad']:
            # mask_img = masks[w]
        # else:
            # mask_img = masks[w,extrap[0]:-extrap[0], extrap[1]:-extrap[1], extrap[2]:-extrap[2]]
        # inds = np.argwhere(mask_img==0) # sorts in order of rows, columns, z, i.e. it iterates through z first, then columns, then rows 
         
        # # Important - we need to make sure we predict in the order of columns, rows, z (otherwise will be conditioning on wrong information) 
        # inds = inds[np.lexsort((inds[:,0],inds[:,2]))]
        # for i in range(len(inds)):
            # x = image2kerasarray(X_pred_same[w:w+1], norm_root)
            # next_X_pred = pixelcnn.model.predict(x, batch_size)
            
            # sampled_pred = next_X_pred[:,inds[i,0],inds[i,1],inds[i,2],:]
            # sampled_pred = sample(sampled_pred[0], temperature=temperature)
            # print("Predicted intensity:",sampled_pred,"Actual intensity:",X_incomplete[w,inds[i,0]+extrap[0],inds[i,1]+extrap[1],inds[i,2]+extrap[2]])
            # X_pred[w,inds[i,0],inds[i,1],inds[i,2]] = sampled_pred # output size image 
            # if model_params['pad']:
                # X_pred_same[w,inds[i,0],inds[i,1],inds[i,2]] = sampled_pred
            # else:
                # X_pred_same[w,inds[i,0]+extrap[0],inds[i,1]+extrap[1],inds[i,2]+extrap[2]] = sampled_pred # input size image 
        # # Since this is a test, we will save as tiffs 
        # print(X_pred[w].shape)
        # X_completed = X_incomplete[w].copy()
        # if not model_params['pad']:
            # X_completed[30:42,30:42,30:42] = X_pred[w][26:38,26:38,26:38] # very specific to the current set up 
        # else:
            # X_completed = X_pred[w]
        # io.writeData(os.path.join(save_dir, 'test_%d_pred.tif'%w),X_completed.astype('uint8'))
        # io.writeData(os.path.join(save_dir, 'test_%d_actual.tif'%w),X_incomplete[w].astype('uint8'))
        # io.writeData(os.path.join(save_dir, 'test_%d_incomp.tif'%w),incomplete_imgs[w].astype('uint8'))

if __name__ == '__main__':
    predict()