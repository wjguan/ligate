import numpy as np
import keras, os 
from keras import backend as K 
from keras.layers import Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D, Activation, Add, add, Lambda, Input, Cropping3D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Model, load_model


class PixelCNN3D(object):
    ''' Keras implementation of vanilla PixelCNN, but in 3D.'''
    def __init__(
        self,
        input_size,
        nb_res_blocks=1,
        nb_filters_h=128,
        nb_filters_d=1024,
        filter_size_1st=(7,7,7),
        filter_size=(3,3,3),
        dropout=False,
        optimizer='adam',
        loss='categorical_crossentropy',
        pad=False,
        es_patience=100,
        save_root=r'results\pixelcnn3d',
        save_best_only=False,
        **kwargs):
        '''
        Args:
            input_size ((int,int,int))       : (height, width, depth) pixels of input images
            nb_res_blocks (int)              : Number of residual blocks (default: 1) 
            nb_filters_h (int)               : Number of filters (equivalent to "h" in the paper). (default:128)
            nb_filters_d (int)               : Number of filters in the layer after residual blocks. (default:128) 
            filter_size_1st ((int, int, int)): Kernel size for the first layer. (default: (7,7,7))
            filter_size ((int, int, int))    : Kernel size for the subsequent layers. (default: (3,3,3))
            dropout (Bool)                   : Whether to use dropout (0.5) or not. (default: False)
            optimizer (str)                  : SGD optimizer (default: 'adam')
            loss (str)                       : Loss function to use (default: 'categorical_crossentropy') 
            pad (bool)                       : Whether to pad the later convolutions (1st one is automatically unpadded) 
            es_patience (int)                : Number of epochs with no improvement after which training will be stopped (EarlyStopping)
            save_root (str)                  : Root directory to which {trained model file, parameter.txt, tensorboard log file} are saved
            save_best_only (bool)            : if True, the latest best model will not be overwritten (default: False)
        '''
        K.set_image_dim_ordering('tf')

        self.input_size = input_size
        self.nb_res_blocks = nb_res_blocks
        self.nb_filters_h = nb_filters_h
        self.nb_filters_d = nb_filters_d 
        self.filter_size_1st = filter_size_1st 
        self.filter_size = filter_size
        self.loss = loss
        self.dropout = dropout 
        self.pad = pad
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.save_best_only = save_best_only

        tensorboard_dir = os.path.join(save_root, 'pixelcnn-tensorboard')
        checkpoint_path = os.path.join(save_root, 'pixelcnn-weights.{epoch:02d}-{val_loss:.4f}.hdf5')
        self.tensorboard = TensorBoard(log_dir=tensorboard_dir)
        ### "save_weights_only=False" causes error when exporting model architecture. (json or yaml)
        self.checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_best_only=save_best_only)
        self.earlystopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0, mode='auto')
    
    @staticmethod
    def _crop_vert(x, filter_size, pad):
        x_shape = K.int_shape(x) 
        # This assumes symmetric x,y image (which should be true) 
        if pad:
            x = Lambda(lambda x: x[:,:x_shape[2],:,:,:])(x)
        else:
            x = Lambda(lambda x: x[:,:x_shape[2],:,filter_size[2]//2:-(filter_size[2]//2),:])(x)
        return x
        
    @staticmethod 
    def _crop_hor(x, filter_size, mask_type, pad):
        ''' shifts after horizontal convolution, only if pad is false. '''
        x_shape = K.int_shape(x) 
        if not pad:
            if mask_type == 'A':
                x = Lambda(lambda x: x[:,filter_size[0]//2:x_shape[1]-filter_size[0]//2,:x_shape[2]-filter_size[1]//2-1,filter_size[2]//2:-(filter_size[2]//2),:])(x)
            else:
                x = Lambda(lambda x: x[:,filter_size[0]//2:x_shape[1]-filter_size[0]//2,:x_shape[2]-filter_size[1]//2,filter_size[2]//2:-(filter_size[2]//2),:])(x)
        else:
            if mask_type == 'A':
                x = Lambda(lambda x: x[:,:,:x_shape[1],:,:])(x)
        return x 
        
    @staticmethod
    def _crop_depth(x, filter_size, pad):
        x_shape = K.int_shape(x)
        if pad:
            x = Lambda(lambda x: x[:,:,:,:-1,:])(x)
        else:
            x = Lambda(lambda x: x[:,:,:,:-filter_size[2]//2,:])(x) 
        return x
            
            
    def _masked_conv(self, x, filter_size, nb_filters, layer_idx, mask_type='B'):
        # Do the vertical part first
        y = x
        if self.pad:
            # if we use padding = same for our layers, then we need to pad differently 
            y = ZeroPadding3D(padding=((filter_size[0]//2, 0), (filter_size[1]//2, filter_size[1]//2), (0,0)), name='v_pad_'+str(layer_idx))(y)
        res = Conv3D(nb_filters, (filter_size[0]//2, filter_size[1], 1), padding='valid', kernel_initializer='he_normal',name='v_conv_'+str(layer_idx))(y)
        res = self._crop_vert(res, filter_size, self.pad) 
        
        # Now do the horizontal part
        y = x
        if self.pad:
            # only need to pad if we are intending to keep the same image size 
            y = ZeroPadding3D(padding=((0, 0), (filter_size[1]//2, 0), (0,0)), name='h_pad_'+str(layer_idx))(y) # only pad on the left 
        if mask_type == 'A':
            y = Conv3D(nb_filters, (1,filter_size[1]//2,1), padding='valid', kernel_initializer='he_normal',name='h_conv_'+str(layer_idx))(y)
        else:
            y = Conv3D(nb_filters, (1,filter_size[1]//2+1,1), padding='valid', kernel_initializer='he_normal',name='h_conv_'+str(layer_idx))(y)
        res_h = self._crop_hor(y, filter_size, mask_type, self.pad) 
        
        # Depth (z)
        y = x
        if self.pad:
            y = ZeroPadding3D(padding=((filter_size[0]//2,filter_size[0]//2),(filter_size[1]//2,filter_size[1]//2),(filter_size[2]//2,0)), name='d_pad_'+str(layer_idx))(y)
        res_d = Conv3D(nb_filters, (filter_size[0], filter_size[1], filter_size[2]//2), padding='valid',kernel_initializer='he_normal',name='d_conv_'+str(layer_idx))(y)
        res_d = self._crop_depth(res_d, filter_size, self.pad) 
        
        # Add these results together to get the total masked convolution 
        res_tot = Add()([res, res_h, res_d])
        return res_tot

    
        
    def _build_layers(self, x):
        ''' Whole architecture of PixelCNN model'''
        
        # Initial 
        y = self._masked_conv(x, self.filter_size_1st, 2*self.nb_filters_h, 0, 'A')
        
        for i in range(0, self.nb_res_blocks):
            z = Activation('relu')(y) 
            if self.dropout:
                z = Dropout(0.5)(z)
            z = Conv3D(self.nb_filters_h, 1, strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='he_normal')(y)
            if self.dropout:
                z = Dropout(0.5)(z)
            z = self._masked_conv(z, self.filter_size, self.nb_filters_h, i+1, 'B')
            z = Activation('relu')(z) 
            if self.dropout:
                z = Dropout(0.5)(z)
            z = Conv3D(2*self.nb_filters_h, 1, strides=(1,1,1), padding='valid', kernel_initializer='he_normal')(z)
            
            
            if not self.pad:
                # will run into problems with size here if don't deal with the cropping. 
                y = Cropping3D(cropping=((self.filter_size[0]//2, self.filter_size[0]//2),
                                         (self.filter_size[1]//2, self.filter_size[1]//2),
                                         (self.filter_size[2]//2, self.filter_size[2]//2)))(y)
            y = Add()([y, z]) 
        
        # FInish up 
        z = Activation('relu')(y)
        if self.dropout:
                z = Dropout(0.5)(z)
        z = Conv3D(self.nb_filters_d, 1, strides=(1,1,1), activation='relu', padding='valid',kernel_initializer='he_normal')(z)
        if self.dropout:
                z = Dropout(0.5)(z)
        z = Conv3D(256, 1, strides=(1,1,1), activation='softmax', padding='valid',kernel_initializer='he_normal')(z)
        
        return z 
        

    def build_model(self):
        ''' build model '''
        input_img = Input(shape=(self.input_size[0], self.input_size[1], self.input_size[2], 1))
        predicted = self._build_layers(input_img)
        self.model = Model(input_img, predicted)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
    

    def fit(
        self,
        x,
        y,
        batch_size,
        nb_epoch,
        validation_data=None,
        shuffle=True):
        ''' call fit function
        Args:
            x (np.ndarray or [np.ndarray, np.ndarray])  : Input data for training
            y (np.ndarray)                              : Label data for training 
            samples_per_epoch (int)                     : Number of data for each epoch
            nb_epoch (int)                              : Number of epoches
            validation_data ((np.ndarray, np.ndarray))  : Validation data
            nb_val_samples (int)                        : Number of data yielded by validation generator
            shuffle (bool)                              : if True, shuffled randomly
        '''
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer, self.earlystopping],
            validation_data=validation_data,
            shuffle=shuffle
        )

    def fit_generator(
        self,
        train_generator,
        samples_per_epoch,
        nb_epoch,
        validation_data=None,
        nb_val_samples=1000):
        ''' call fit_generator function
        Args:
            train_generator (object)        : image generator built by "build_generator" function
            samples_per_epoch (int)         : Number of data for each epoch
            nb_epoch (int)                  : Number of epoches
            validation_data (object/array)  : generator object or numpy.ndarray
            nb_val_samples (int)            : Number of data yielded by validation generator
        '''
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=samples_per_epoch,
            epochs=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer, self.earlystopping],
            validation_data=validation_data,
            validation_steps=nb_val_samples
        )


    def load_model(self, checkpoint_file):
        ''' restore model from checkpoint file (.hdf5) '''
        self.model = load_model(checkpoint_file)

    def export_to_json(self, save_root):
        ''' export model architecture config to json file '''
        with open(os.path.join(save_root, 'pixelcnn_model.json'), 'w') as f:
            f.write(self.model.to_json())

    def export_to_yaml(self, save_root):
        ''' export model architecture config to yaml file '''
        with open(os.path.join(save_root, 'pixelcnn_model.yml'), 'w') as f:
            f.write(self.model.to_yaml())


    @classmethod
    def predict(self, x, batch_size):
        ''' generate image pixel by pixel
        Args:
            x (x: numpy.ndarray : x = input image)
            batch_size (int) - batch_size for prediction 
        Returns:
            predict (numpy.ndarray)        : generated image
        '''
        
        return self.model.predict(x, batch_size)


    def print_train_parameters(self, save_root):
        ''' print parameter list file '''
        print('\n########## PixelCNN options ##########')
        print('input_size\t: %s' % (self.input_size,))
        print('nb_res_blocks\t: %s' % self.nb_res_blocks)
        print('nb_filters_h\t: %s' % self.nb_filters_h)
        print('nb_filters_d\t: %s' % self.nb_filters_d) 
        print('filter_size_1st\t: %s' % (self.filter_size_1st,))
        print('filter_size\t: %s' % (self.filter_size,))
        print('pad\t\t: %s' % self.pad)
        print('dropout\t: %s\n' %self.dropout)
        print('optimizer\t: %s' % self.optimizer)
        print('loss\t\t: %s' % self.loss)
        print('es_patience\t: %s' % self.es_patience)
        print('save_root\t: %s' % save_root)
        print('save_best_only\t: %s' % self.save_best_only)
        print('\n')

    def export_train_parameters(self, save_root):
        ''' export parameter list file '''
        with open(os.path.join(save_root, 'parameters.txt'), 'w') as txt_file:
            txt_file.write('########## PixelCNN options ##########\n')
            txt_file.write('input_size\t: %s\n' % (self.input_size,))
            txt_file.write('nb_res_blocks\t: %s\n' % self.nb_res_blocks)
            txt_file.write('nb_filters_h\t: %s\n' % self.nb_filters_h)
            txt_file.write('nb_filters_d\t: %s\n' % self.nb_filters_d)
            txt_file.write('filter_size_1st\t: %s\n' % (self.filter_size_1st,))
            txt_file.write('filter_size\t: %s\n' % (self.filter_size,))
            txt_file.write('pad\t\t: %s\n' % self.pad)
            txt_file.write('dropout\t: %s\n' %self.dropout)
            txt_file.write('optimizer\t: %s\n' % self.optimizer)
            txt_file.write('loss\t\t: %s\n' % self.loss)
            txt_file.write('es_patience\t: %s\n' % self.es_patience)
            txt_file.write('save_root\t: %s\n' % save_root)
            txt_file.write('save_best_only\t: %s\n' % self.save_best_only)
            txt_file.write('\n')
 
class GatedPixelCNN3D(object): 
    pass
    
class BidirectionalPixelCNN3D(object):
    ''' Bidirectional PixelCNN '''
    pass 
