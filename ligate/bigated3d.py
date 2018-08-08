import numpy as np
import keras, os 
from keras import backend as K 
from keras.layers import Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D, Activation, Add, Multiply, Lambda, Input, Cropping3D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Model, load_model
from ligate.utils import discretized_mix_logistic_loss
from ligate.gated3d import GatedCNN, GatedPixelCNN3D
import tensorflow as tf

class BiGatedPixelCNN3D(object):
    ''' Bidirectional gated pixel CNN.'''
    def __init__(
        self,
        input_size,
        nb_res_blocks=1,
        nb_filters=128,
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
            dropout (bool)                   : Whether to add Dropout at 0.5 (default: False) 
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
        self.nb_filters = nb_filters 
        self.filter_size_1st = filter_size_1st 
        self.filter_size = filter_size
        self.loss = loss
        self.dropout = dropout
        self.pad = pad
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.save_best_only = save_best_only
        self.save_root = save_root

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
            
            
    def _masked_conv(self, y, filter_size, nb_filters, stack_name, layer_idx, mask_type='B'):
        # Vertical
        if stack_name == 'vertical':
            if self.pad:
                # if we use padding = same for our layers, then we need to pad differently 
                y = ZeroPadding3D(padding=((filter_size[0]//2, 0), (filter_size[1]//2, filter_size[1]//2), (0,0)), name='v_pad_'+str(layer_idx))(y)
            res = Conv3D(nb_filters, (filter_size[0]//2, filter_size[1], 1), padding='valid', kernel_initializer='he_normal',name='v_conv_'+str(layer_idx))(y)
            res = self._crop_vert(res, filter_size, self.pad) 
        
        # Horizontal
        elif stack_name == 'horizontal':
            if self.pad:
                # only need to pad if we are intending to keep the same image size 
                y = ZeroPadding3D(padding=((0, 0), (filter_size[1]//2, 0), (0,0)), name='h_pad_'+str(layer_idx))(y) # only pad on the left 
            if mask_type == 'A':
                y = Conv3D(nb_filters, (1,filter_size[1]//2,1), padding='valid', kernel_initializer='he_normal',name='h_conv_'+str(layer_idx))(y)
            else:
                y = Conv3D(nb_filters, (1,filter_size[1]//2+1,1), padding='valid', kernel_initializer='he_normal',name='h_conv_'+str(layer_idx))(y)
            res = self._crop_hor(y, filter_size, mask_type, self.pad) 
        
        # Depth (z)
        elif stack_name == 'depth':
            if self.pad:
                y = ZeroPadding3D(padding=((filter_size[0]//2,filter_size[0]//2),(filter_size[1]//2,filter_size[1]//2),(filter_size[2]//2,0)), name='d_pad_'+str(layer_idx))(y)
            res_d = Conv3D(nb_filters, (filter_size[0], filter_size[1], filter_size[2]//2), padding='valid',kernel_initializer='he_normal',name='d_conv_'+str(layer_idx))(y)
            res = self._crop_depth(res_d, filter_size, self.pad) 
        
        return res
    
        
    def build_model(self):    
        ## Build the reverse model first. 
        if self.pad:
            nb_channels = 1
        else:
            nb_channels = 1
        reverse_input = Input(shape=(self.input_size[0], self.input_size[1], self.input_size[2],nb_channels+1))
        mask = Lambda(lambda x: x[:,:,:,:,-1:])(reverse_input) # get the context mask 
        reverse_net = GatedPixelCNN3D(input_size=self.input_size,nb_res_blocks=self.nb_res_blocks,nb_filters_h=self.nb_filters,
                                    nb_filters_d=self.nb_filters,filter_size_1st=self.filter_size_1st,filter_size=self.filter_size,
                                    dropout=self.dropout,optimizer=self.optimizer,loss=None,es_patience=self.es_patience,
                                    save_root=self.save_root,pad=self.pad,save_best_only=self.save_best_only)
        x = reverse_net.build_layers(reverse_input) # loss=None ensures we get it before the final activation.
        
        ## Foward model 
        input_img = Input(shape=(self.input_size[0], self.input_size[1], self.input_size[2],nb_channels))
        z = self._masked_conv(input_img, self.filter_size_1st, 2*self.nb_filters, 'depth', 10)
        # to be fed into vertical 
        z_feed = Conv3D(2*self.nb_filters, 1, strides=(1,1,1), kernel_initializer='he_normal')(z)
        z_out = GatedCNN(self.nb_filters)(z)
        if self.dropout:
            z_out = Dropout(0.5)(z_out)
        
        v = self._masked_conv(x, self.filter_size_1st, 2*self.nb_filters, 'vertical', 10)
        # to be fed into horizontal 
        v_out, v_feed = GatedCNN(self.nb_filters, z_map=z_feed)(v)
        if self.dropout:
            v_out = Dropout(0.5)(v_out)
        
        h = self._masked_conv(x, self.filter_size_1st, 2*self.nb_filters, 'horizontal', 10, mask_type='A')
        h_out = GatedCNN(self.nb_filters, v_map=v_feed)(h)
        h_out = Conv3D(self.nb_filters, 1, kernel_initializer='he_normal')(h_out)
        if self.dropout:
            h_out = Dropout(0.5)(h_out)
        
        # Combine with the reverse PixelCNN 
        # need to actually reverse the direction 
        x = Lambda(lambda o: K.reverse(o,axes=[2,1,0]))(x)
        reverse = Conv3D(self.nb_filters, 1, kernel_initializer='he_normal')(x)
        h_out = Add()([reverse, h_out])
        h_out = Lambda(lambda r: K.relu(r))(h_out)
        
        for i in range(0, self.nb_res_blocks):
            z = self._masked_conv(z_out, self.filter_size, 2*self.nb_filters, 'depth', i+11)
            z_feed = Conv3D(2*self.nb_filters, 1, strides=(1,1,1), kernel_initializer='he_normal')(z)
            if i != self.nb_res_blocks-1:
                z_out = GatedCNN(self.nb_filters)(z)
                if self.dropout:
                    z_out = Dropout(0.5)(z_out)
            
            v = self._masked_conv(v_out, self.filter_size, 2*self.nb_filters, 'vertical', i+11)
            v_out, v_feed = GatedCNN(self.nb_filters, z_map=z_feed)(v)
            if self.dropout and i != self.nb_res_blocks-1:
                v_out = Dropout(0.5)(v_out)
                
            h_out_prev = h_out
            h = self._masked_conv(h_out, self.filter_size_1st, 2*self.nb_filters, 'horizontal', i+11)
            h_out = GatedCNN(self.nb_filters, v_map=v_feed)(h)
            h_out = Conv3D(self.nb_filters, 1, kernel_initializer='he_normal')(h_out)
            if self.dropout:
                h_out = Dropout(0.5)(h_out)
            
            if not self.pad:
                # will run into problems with size here if don't deal with the cropping. 
                h_out_prev = Cropping3D(cropping=((self.filter_size[0]//2, self.filter_size[0]//2),
                                         (self.filter_size[1]//2, self.filter_size[1]//2),
                                         (self.filter_size[2]//2, self.filter_size[2]//2)))(h_out_prev)
            
            reverse = Conv3D(self.nb_filters, 1, kernel_initializer='he_normal')(x)
            h_out = Add()([reverse, h_out]) # reverse 
            h_out = Lambda(lambda r: K.relu(r))(h_out) # use relu nonlinearity 
            h_out = Add()([h_out_prev, h_out]) 
        
        # FInish up 
        h = Conv3D(self.nb_filters, 1, strides=(1,1,1), activation='relu', padding='valid',kernel_initializer='he_normal')(h_out)
        if self.loss == 'discretized_mix_logistic_loss':
            h = Conv3D(30, 1, strides=(1,1,1), activation=None, padding='valid',kernel_initializer='he_normal')(h)
        elif self.loss == 'categorical_crossentropy':
            h = Conv3D(256, 1, strides=(1,1,1), activation='softmax', padding='valid',kernel_initializer='he_normal')(h)
            # h = Lambda(lambda r: (1.0-r[1])*r[0])([h, mask]) # only condition on missing pixels 
        self.model = Model([input_img, reverse_input], h) 
        if self.loss == 'discretized_mix_logistic_loss':
            self.model.compile(optimizer=self.optimizer, loss=discretized_mix_logistic_loss)
        else:
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        
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
    
    def print_train_parameters(self, save_root):
        ''' print parameter list file '''
        print('\n########## PixelCNN options ##########')
        print('input_size\t: %s' % (self.input_size,))
        print('nb_res_blocks\t: %s' % self.nb_res_blocks)
        print('nb_filters\t: %s' % self.nb_filters)
        print('filter_size_1st\t: %s' % (self.filter_size_1st,))
        print('filter_size\t: %s' % (self.filter_size,))
        print('pad\t\t: %s' % self.pad)
        print('dropout\t\t: %s' % self.dropout)
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
            txt_file.write('nb_filters\t: %s\n' % self.nb_filters)
            txt_file.write('filter_size_1st\t: %s\n' % (self.filter_size_1st,))
            txt_file.write('filter_size\t: %s\n' % (self.filter_size,))
            txt_file.write('pad\t\t: %s\n' % self.pad)
            txt_file.write('dropout\t\t: %s\n' % self.dropout)
            txt_file.write('optimizer\t: %s\n' % self.optimizer)
            txt_file.write('loss\t\t: %s\n' % self.loss)
            txt_file.write('es_patience\t: %s\n' % self.es_patience)
            txt_file.write('save_root\t: %s\n' % save_root)
            txt_file.write('save_best_only\t: %s\n' % self.save_best_only)
            txt_file.write('\n')