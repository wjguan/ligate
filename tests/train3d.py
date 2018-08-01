import argparse, os, zarr, sys, glob
import numpy as np
from ligate.utils import * 
from distutils.util import strtobool

def train():
    ## Set GPU using tf 
    import tensorflow as tf 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    from keras.backend.tensorflow_backend import set_session
    set_session(tf.Session(config=config))
    
    
    parser = argparse.ArgumentParser(description='Parse model parameters.')
    parser.add_argument("--model_type", help="Gated or vanilla [Required]", type=str, metavar="INT")
    parser.add_argument("--nb_epoch", help="Number of epochs [Required]", type=int, metavar="INT")
    parser.add_argument("--batch_size", help="Minibatch size [Required]", type=int, metavar="INT")
    parser.add_argument("--nb_res_blocks", help="Number of residual blocks. (default: 1)", metavar="INT")
    parser.add_argument("--nb_filters_h", help="Number of filters for each layer (or double this amount). (default: 128)", metavar="INT")
    parser.add_argument("--nb_filters_d", help="Number of filters for penultimate layers. (default: 1024)", metavar="INT")
    parser.add_argument("--filter_size_1st", help="Filter size for the first layer. (default: (7,7,7))", metavar="INT,INT,INT")
    parser.add_argument("--filter_size", help="Filter size for the subsequent layers. (default: (3,3,3))", metavar="INT,INT,INT")
    parser.add_argument("--optimizer", help="SGD optimizer (default: adam)", type=str, metavar="OPT_NAME")
    parser.add_argument("--dropout", help="Whether to use dropout of 0.5 or not (default: False)", type=str, metavar="BOOL")
    parser.add_argument("--pad", help="Whether to pad images or not (default: False)", type=str, metavar="BOOL")
    parser.add_argument("--loss", help="Loss function (default: categorical_crossentropy)", type=str, metavar="OPT_NAME")
    parser.add_argument("--es_patience", help="Patience parameter for EarlyStopping (default: 100)", type=int, metavar="INT")
    parser.add_argument("--save_root", help="Root directory which trained files are saved (default: results\pixelcnn3d)", type=str, metavar="DIR_PATH")
    parser.add_argument("--load_data_root", help="Root directory which training/validation data are loaded in from. (default: data\lectin594)", type=str)
    parser.add_argument("--save_best_only", help="The latest best model will not be overwritten (default: False)", type=str, metavar="BOOL")
    parser.add_argument("--random_rotation", help="Whether to add random rotations to the data. (default: True)", type=str, metavar="BOOL")
    parser.add_argument("--random_flip", help="Whether to add random flips to the data. (default: True)", type=str, metavar="BOOL") 
    parser.add_argument("--data_type", help="Data type of the images. (default: uint8)", type=str, metavar="DTYPE")
    parser.add_argument("--target_size", help="Target shape of the model. (default: (42,42,42))", metavar="INT,INT,INT")
    parser.add_argument("--continue_training", help="Whether to load from existing checkpoint file and continue training. (default: False)", metavar="BOOL")
    #parser.add_argument("--plot_model", help="If True, plot a Keras model (using graphviz)", type=str, metavar="BOOL")
    
    args = parser.parse_args() 
    
    if args.model_type == 'vanilla':
        from ligate.models import PixelCNN3D
        print("it's not gated")
    elif args.model_type == 'gated':
        from ligate.gated3d import GatedPixelCNN3D as PixelCNN3D 
        print("it's gated")
    
    ## Load data 
    if args.load_data_root:
        load_data_root = args.load_data_root 
    else:
        load_data_root = r'data\lectin594'
    
    X_train = zarr.open(os.path.join(load_data_root, 'X_train.zarr'), mode='r')
    X_val = zarr.open(os.path.join(load_data_root, 'X_val.zarr'), mode='r') 
    
    
    ### build PixelCNN model ###
    model_params = {}
    model_params['input_size'] = X_train.shape[1:] 
    if args.save_root:
        model_params['save_root'] = args.save_root 
    if args.nb_res_blocks:
        model_params['nb_res_blocks'] = int(args.nb_res_blocks)
    if args.nb_filters_h:
        model_params['nb_filters_h'] = int(args.nb_filters_h)
    if args.nb_filters_d:
        model_params['nb_filters_d'] = int(args.nb_filters_d)
    if args.filter_size_1st:
        model_params['filter_size_1st'] = tuple(map(int, args.filter_size_1st.split(',')))
    if args.filter_size:
        model_params['filter_size'] = tuple(map(int, args.filter_size.split(',')))
    if args.optimizer:
        model_params['optimizer'] = args.optimizer
    if args.dropout:
        model_params['dropout'] = args.dropout
    if args.loss:
        model_params['loss'] = args.loss
    if args.es_patience:
        model_params['es_patience'] = int(args.patience)
    if args.save_best_only:
        model_params['save_best_only'] = strtobool(args.save_best_only)
    if args.pad:
        model_params['pad'] = strtobool(args.pad)
    
    save_root = args.save_root if args.save_root else r'results\pixelcnn3d'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    try:
        nb_epoch = int(args.nb_epoch)
        batch_size = int(args.batch_size)
    except:
        sys.exit("Error: {--nb_epoch, --batch_size} must be specified.")
    
    continue_training = strtobool(args.continue_training) if args.continue_training else False 
    pixelcnn = PixelCNN3D(**model_params)
    pixelcnn.build_model()
    if continue_training:
        last_filename = glob.glob(save_root + r'\*.hdf5')[-1]
        pixelcnn.model.load_weights(last_filename)
    pixelcnn.model.summary() 
    
    # Print and export the parameters being used 
    pixelcnn.print_train_parameters(save_root)
    pixelcnn.export_train_parameters(save_root)
    
    
    ## Train the model using a generator 
    random_rotation = strtobool(args.random_rotation) if args.random_rotation else True 
    random_flip = strtobool(args.random_flip) if args.random_flip else True 
    target_size = tuple(map(int, args.target_size.split(','))) if args.target_size else (42,42,42) 
    dtype = args.data_type if args.data_type else 'uint8'
    if not os.path.isfile(os.path.join(load_data_root, 'mean.npy')):
        _,_ = normalize(os.path.join(load_data_root, 'X_train.zarr')) 
        
    if model_params['loss'] == 'discretized_mix_logistic_loss':
        train_generator = data_generator(X_train, target_size, batch_size, load_data_root, rescale=True, random_rotation=random_rotation, random_flip=random_flip, dtype=dtype)
        validation_generator = data_generator(X_val, target_size, batch_size, load_data_root, rescale=True, random_rotation=random_rotation, random_flip=random_flip, dtype=dtype) 
    else:
        train_generator = data_generator(X_train, target_size, batch_size, load_data_root, rescale=False, random_rotation=random_rotation, random_flip=random_flip, dtype=dtype)
        validation_generator = data_generator(X_val, target_size, batch_size, load_data_root, rescale=False, random_rotation=random_rotation, random_flip=random_flip, dtype=dtype)     
    pixelcnn.fit_generator(train_generator=train_generator, 
                                samples_per_epoch=int(np.ceil(len(X_train)/batch_size)), 
                                nb_epoch=nb_epoch, 
                                validation_data=validation_generator,
                                nb_val_samples=int(np.ceil(len(X_val)/batch_size)))
    
    
    
    
if __name__ == '__main__':
    train()