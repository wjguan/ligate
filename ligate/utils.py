import numpy as np
import ligate.IO as io 
from keras.utils import to_categorical
import time, zarr, os, random
from sklearn.model_selection import train_test_split 
import tensorflow as tf 
from keras import backend as K

# Contains neural network and preprocessing tools 


def make_image_chunks_3d(img, desired_shape, overlap, save_dir):
    ''' 
    Makes 3D image chunks based on how much overlap is required (depending on padding and architecture)
    For now, don't worry about large data 
    
    Inputs:
    img - 3D array containing the image to be chunked 
    desired_shape - (x,y,z) tuple of the desired shape of the chunks 
    overlap - (x,y,z) tuple of the desired pixel overlap between chunks. 
            It is assumed that the padding on edge cases will be half of that. 
    save_dir - if not None, then is the directory in which to save the chunks. If None, don't save 
    
    Outputs:
    img_chunks - if save_dir is None, then all of the img chunks will be returned, otherwise we save to the save_dir 
    
    '''
    padded_img = np.pad(img, ((overlap[0]//2,overlap[0]//2),(overlap[1]//2,overlap[1]//2),(overlap[2]//2,overlap[2]//2)), 'constant')
    
    # Compute the start points for each chunk for each dimension 
    extra_padding = []
    start_points = [] 
    for i in range(3):
        imgdim = padded_img.shape[i]
        dim = desired_shape[i]
        ovdim = overlap[i]
        
        # get the amount of extra padding needed and start points 
        if imgdim % (dim-ovdim) > ovdim:
            extra = imgdim//(dim-ovdim)*(dim-ovdim) + dim - imgdim
            start = np.linspace(0, imgdim//(dim-ovdim) * (dim-ovdim), imgdim//(dim-ovdim)+1, dtype='uint16')
        else:
            extra = (imgdim//(dim-ovdim)-1)*(dim-ovdim)+dim - imgdim
            start = np.linspace(0, (imgdim//(dim-ovdim)-1) * (dim-ovdim), imgdim//(dim-ovdim), dtype='uint16')
        extra_padding.append(extra)
        start_points.append(start)
    
    # Add extra 0's at end so that we have just the right dimensions. 
    padded_img = np.pad(padded_img, ((0, extra_padding[0]),(0, extra_padding[1]),(0, extra_padding[2])), 'constant') 
    
    print(start_points[0])
    print(start_points[1])
    print(start_points[2])
    
    if save_dir is None:
        img_chunks = np.zeros((len(start_points[0]), len(start_points[1]), len(start_points[2]), desired_shape[0], desired_shape[1], desired_shape[2]),dtype='uint16')
    else:
        img_chunks = zarr.open(save_dir, mode='w', 
                               shape=(len(start_points[0]), len(start_points[1]), len(start_points[2]), desired_shape[0], desired_shape[1], desired_shape[2]),
                               #chunks=(len(start_points[0]), len(start_points[1]), len(start_points[2])), 
                               dtype='i4',
                               synchronizer=zarr.ProcessSynchronizer(os.path.join(os.path.dirname(save_dir),r'temp.sync')))
    for i in range(len(start_points[0])):
        for j in range(len(start_points[1])):
            for k in range(len(start_points[2])):
                ii = start_points[0][i]
                jj = start_points[1][j]
                kk = start_points[2][k]
                img_chunks[i,j,k,:,:,:] = padded_img[ii:ii+desired_shape[0],jj:jj+desired_shape[1],kk:kk+desired_shape[2]]
    
    return img_chunks 
                    
def make_image_chunks_3d_nopad(img, desired_shape, overlap, save_dir):
    '''
    Makes image chunks, but without padding the image at all. 
    This ensures that we only use information that is available from the data. 
    
    '''
    pass 
    
def stitch_image_chunks():
    '''
    Stitches image chunks back together into original image.
    '''
    
    pass
    
def normalize(train_data_root, save_root=None):
    '''
    Computes the normalization constants (mean, standard devitaion) of the training set and then 
    
    
    Inputs:
    data_root (str) - where the training data (zarr file) is located
    save_root (str) - where the mean.npy and std.npy files should be stored. (default=None) 
    
    Outputs:
    mean_px (float) - mean of the training data 
    std_px (float) - std deviation of the training data 
    '''
    
    if save_root is None:
        save_root = os.path.dirname(train_data_root)
    X_train = zarr.load(train_data_root)
    mean_px = np.mean(X_train) 
    std_px = np.std(X_train) 
    np.save(os.path.join(save_root, 'mean.npy'), mean_px)
    np.save(os.path.join(save_root, 'std.npy'), std_px) 
    return mean_px, std_px 

def load_train_val_data(X, test_size, save_dir, manual_shuffle=False):
    '''
    From existing zarr file, we create new zarr files for the training and validation sets. 
    
    Inputs:
    X (zarr) - zarr array containing all of the data 
    test_size (float) - the fraction of samples to be used for the validatoin set 
    save_dir (str) - the directory under which we will save the training and validation zarr files 
    manual_shuffle (bool) - if True, then we shuffle the indices manually (this reduces memory strain). (default: False) 
    
    Outputs:
    X_train (zarr) - zarr array containing the training samples 
    X_val (zarr) - zarr array containing the validation samples 
    '''
    total_num = np.prod(X.shape[:len(X.shape)//2])
    if manual_shuffle:
        if len(X.shape)//2 == 3:
            train_shape = (int(np.floor((1-test_size)*total_num)), X.shape[3], X.shape[4], X.shape[5])
            val_shape = (int(np.ceil(test_size*total_num)), X.shape[3], X.shape[4], X.shape[5])
        elif len(X.shape)//2 == 2:
            train_shape = (int(np.floor((1-test_size)*total_num)), X.shape[2], X.shape[3])
            val_shape = (int(np.ceil(test_size*total_num)), X.shape[2], X.shape[3])
            
        X_train = zarr.open(os.path.join(save_dir, 'X_train.zarr'), mode='w', 
                               shape=train_shape,
                               dtype='i4',
                               synchronizer=zarr.ProcessSynchronizer(os.path.join(os.path.dirname(save_dir),r'temp.sync')))
        X_val = zarr.open(os.path.join(save_dir, 'X_val.zarr'), mode='w', 
                               shape=val_shape,
                               dtype='i4',
                               synchronizer=zarr.ProcessSynchronizer(os.path.join(os.path.dirname(save_dir),r'temp_val.sync')))
        shuffled_index = list(range(total_num))
        random.shuffle(shuffled_index)
        
        j = 0 # keeps track of when to switch over to filling the validation array 
        for i in shuffled_index:
            if j < train_shape[0]:
                if len(X.shape)//2 == 3:
                    ii,jj,kk = np.unravel_index(i, X.shape[:len(X.shape)//2])
                    X_train[j % train_shape[0]] = X[ii,jj,kk]
                elif len(X.shape)//2 == 2:
                    ii,jj = np.unravel_index(i, X.shape[:len(X.shape)//2])
                    X_train[j % train_shape[0]] = X[ii,jj]
            else:
                if len(X.shape)//2 == 3:
                    ii,jj,kk = np.unravel_index(i, X.shape[:len(X.shape)//2])
                    X_val[(j-train_shape[0]) % val_shape[0]] = X[ii,jj,kk]
                elif len(X.shape)//2 == 2:
                    ii,jj = np.unravel_index(i, X.shape[:len(X.shape)//2])
                    X_val[(j-train_shape[0]) % val_shape[0]] = X[ii,jj]
            j += 1
    else:
        X_all = X[:]
        if len(X.shape)//2 == 3:
            X_all = np.reshape(X_all, (total_num, X_all.shape[3], X_all.shape[4], X_all.shape[5]))
        elif len(X.shape)//2 == 2:
            X_all = np.reshape(X_all, (total_num, X_all.shape[2], X_all.shape[3]))
        X_train, X_val = train_test_split(X_all, test_size=test_size, shuffle=True) 
        if save_dir is not None:
            zarr.save(os.path.join(save_dir, 'X_train.zarr'), X_train)
            zarr.save(os.path.join(save_dir, 'X_val.zarr'), X_val) 
    return X_train, X_val 
    
    
def image2kerasarray(img, save_root):
    '''
    Converts an array (img) into a keras-processable array by normalizing and adding a "num_channels" dim.
    
    Inputs:
    img - nD array to be reshaped
    save_root (str) - directory where the mean and standard deviation are located for normalization 
    '''
    mean_px = np.load(os.path.join(save_root, 'mean.npy')) 
    std_px = np.load(os.path.join(save_root, 'std.npy'))
    img = (img - mean_px.astype('float32')) / std_px.astype('float32')
    
            
    return np.expand_dims(img, axis=-1)
    
    
def _image2labelmap(img, target_size, dtype='uint8'):
    '''
    Converts an image to the appropriate label map for training the output 
    e.g. if we are predicting the 8-bit pixel intensities of images, then each array would be 
    made into 8-bit (0-255). 
    
    Inputs:
    img (numpy ndarray) - the arrays for which we are converting 
    target_size (int, int, int) - 
    dtype (str) - the data type of the output 
    
    Outputs:
    y (int,int,int, 2^num_bits) - the output array with one-hot vectors for each of the pixels 
    '''
    num_bits = int(dtype[-1])
    image = img.astype(dtype) 
    y = to_categorical(image, num_classes=2**num_bits)
    y = y.astype('bool')
    if img.shape[0]-target_size[0] > 0:
        border_size = ((img.shape[0]-target_size[0])//2, (img.shape[1]-target_size[1])//2, (img.shape[2]-target_size[2])//2)
        y = y[border_size[0]:-border_size[0], border_size[1]:-border_size[1], border_size[2]:-border_size[2],:]
    
    return y

def rescale_image(img, bounds=[-1,1], dtype='uint8'):
    '''
    Rescales an image to be between bounds.
    
    Inputs: 
    img (numpy ndarray) - image to rescale
    bounds (list) - the bounds to rescale the image to (usually 0,1 or -1,1)
    dtype (str) - type of data to rescale 
    
    Outputs:
    rescaled_img (numpy ndarray, float32) - rescaled image 
    '''
    # if dtype=='uint8':
        # max_val = 255
    # elif dtype=='uint16':
        # max_val = 65535
    img = img.astype('uint8')
    max_val=255.0
    
    difference = bounds[1]-bounds[0]
    return ((img / max_val) * difference + bounds[0]).astype('float32') 
    
    
    
def data_generator(X, target_size, batch_size, save_root, rescale=False, random_rotation=True, random_flip=True, dtype='uint8'):
    '''
    A generator for producing images from a zarr directory. 
    
    Inputs: 
    X (zarr or numpy array) - containing all of the training images 
    target_size (int, int, int) - the target shape of the arrays 
    batch_size (int) - size of the minibatch 
    save_root (str) - directory where mean.npy and std.npy are stored for normalization 
    rescale (bool) - if True (use for discretized logistic mixture loss), then rescale it to be between -1 and 1 
    random_rotation (bool) - if True, will randomly rotate the image 90 degrees in any direction (default: True) 
    random_flip (bool) - if True, will randomly flip the image across any axis (default: True) 
    dtype (str) - the data type of the labels (default: uint8) 
    
    Outputs:
    (x,y) - the batch of training images and associated "labels" 
    '''
    x = np.zeros((batch_size, X.shape[1], X.shape[2], X.shape[3], 1))
    if rescale:
        num_channels=1
    else:
        if isinstance(dtype, str):
            num_channels = 2**int(dtype[-1])
    y = np.zeros((batch_size, target_size[0], target_size[1], target_size[2], num_channels))
    
    while True:
        batch_idx = 0
        shuffled_index = list(range(len(X)))
        random.shuffle(shuffled_index)

        for i in shuffled_index:
            if random_rotation:
                k = np.random.randint(4, size=1)
                axes = tuple(np.random.choice(range(len(X.shape)-1), size=2, replace=False)) 
                new_arr = np.rot90(X[i], k=k, axes=axes)
            else:
                new_arr = X[i]
            flip_axis = random_flip*np.random.randint(len(X.shape), size=1) # is 0 if we don't want to flip 
            if flip_axis > 0:
                # print(flip_axis)
                # print(type(new_arr), new_arr.shape)
                new_arr = np.flip(new_arr, axis=flip_axis[0]-1)
            x[batch_idx % batch_size] = image2kerasarray(new_arr, save_root)
            if rescale:
                scaled_img = rescale_image(new_arr, bounds=[-1,1], dtype=str(new_arr.dtype))
                y[batch_idx % batch_size] = np.expand_dims(scaled_img, axis=-1)
            else:
                y[batch_idx % batch_size] = _image2labelmap(new_arr, target_size, dtype=dtype)
            batch_idx += 1
            if (batch_idx % batch_size) == 0:
                yield (x, y)
                
                
def sample(preds, temperature=1.0):
        '''
        helper function to sample an index from a probability array
        
        Inputs:
        preds (list of floats) - prediction probabilities for each given pixel intensity 
        temperature (float) - the log probabilities are scaled to this value 
        
        Outputs:
        (float) - value drawn from a multinomial distribution between 0 and 1 
        '''
        preds = np.asarray(preds).astype('float32')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        choices = range(len(preds))
        return np.random.choice(choices, p=preds)
        
        # probas = np.random.multinomial(1, preds, 1)
        # return np.argmax(probas).astype('uint8')

        
def _int_shape(x):
    return list(map(int, x.get_shape()))
    
def _log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis)) 

def _log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))
    
def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # xs = _int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,50,50,50,1)
    # ls = _int_shape(l) # predicted distribution, e.g. (B,50,50,50,30)
    
    batch_size = tf.shape(l)[0] # gives us the correct variable batch size, otherwise tf.reshape won't work 
    ls = l.get_shape().as_list()
    xs = ls[:-1] + [1] 
    nr_mix = int(ls[-1] / 3) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,:,nr_mix:], [batch_size] + xs[1:] + [nr_mix*2])
    means = l[:,:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,:,nr_mix:2*nr_mix], -7.)
    
    x = tf.reshape(x, [batch_size] + xs[1:] + [1]) + tf.zeros([batch_size] + xs[1:] + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + _log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(_log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(_log_sum_exp(log_probs),[1,2])
    
    
    
def _unrescale_image(img, bounds=[-1,1], dtype='uint8'):
    '''
    Re-Rescales an image from some bounds back to the original data type. 
    
    Inputs: 
    img (numpy ndarray) - image or array to rescale that's within bounds 
    bounds (list) - the bounds from which to rescale the image back (usually 0,1 or -1,1)
    dtype (str) - type of data to rescale 
    
    Outputs:
    rescaled_img (numpy ndarray, float32) - rescaled image 
    '''
    max_val=255.0
    difference = bounds[1] - bounds[0] 
    new_img = (img - bounds[0])/difference * max_val  
    return new_img.astype(dtype) 
    

## TODO: Modify the following function 
def sample_from_discretized_mix_logistic(l,nr_mix=10):
    l = tf.convert_to_tensor(l)
    ls = l.get_shape().as_list()
    xs = ls[:-1] + [1]
    
    # need to write this for one pixel being predicted
    logit_probs = l[:,:nr_mix]
    l = tf.reshape(l[:,nr_mix:],xs+[nr_mix*2])
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 1), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    means = tf.reduce_sum(l[:,:,:nr_mix]*sel, 2)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,nr_mix:2*nr_mix]*sel,2), -7.)
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1.-1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1.-u))
    x0 = tf.minimum(tf.maximum(x[:,0],-1.),1.)
    
    ## The following is for whole images...
    # # unpack parameters
    # logit_probs = l[:, :, :, :, :nr_mix]
    # l = tf.reshape(l[:, :, :, :, nr_mix:], xs + [nr_mix*2])
    # # sample mixture indicator from softmax
    # sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 4), depth=nr_mix, dtype=tf.float32)
    # sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # # select logistic parameters
    # means = tf.reduce_sum(l[:,:,:,:,:,:nr_mix]*sel,5)
    # log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,:,nr_mix:2*nr_mix]*sel,5), -7.)
    # # sample from logistic & clip to interval
    # # we don't actually round to the nearest 8bit value when sampling
    # u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    # x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    # x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    
    ## Now we have to convert back to 0-255 range 
    x0 = K.eval(x0)  # convert back to numpy array 
    sampled = _unrescale_image(x0)
    return sampled
    
    