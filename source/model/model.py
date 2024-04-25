
from tensorflow.keras.layers import Conv3D, MaxPool3D, concatenate, Input, Dropout, PReLU, Conv3DTranspose, BatchNormalization, SpatialDropout3D, Conv2D, Add, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model
# from keras_contrib.layers import CRF
from model.ConcreteDropout.spatialConcreteDropout import Spatial3DConcreteDropout

##### U-Net

def unet3d(patch_size, n_label):

    input_layer = Input(shape=patch_size)
    
    d1 = unet_core(input_layer, filter_size=64, kernel_size=(3, 3, 3))
    l = Conv3D(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d1)
    d2 = unet_core(l, filter_size=128, kernel_size=(3, 3, 3))
    l =  Conv3D(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d2)
    d3 = unet_core_dropout(l, filter_size=256, kernel_size=(3, 3, 3))
    l = Conv3D(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d3)
    b = unet_core_dropout(l, filter_size=512, kernel_size=(3, 3, 3))

    l = Conv3DTranspose(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(b)
    l = concatenate([l, d3], axis=-1)
    u3 = unet_core_dropout(l, filter_size=256, kernel_size=(3, 3, 3))
    l = Conv3DTranspose(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u3)
    l = concatenate([l, d2], axis=-1)
    u2 = unet_core(l, filter_size=128, kernel_size=(3, 3, 3))
    l = Conv3DTranspose(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u2)
    l = concatenate([l, d1], axis=-1)
    u1 = unet_core(l, filter_size=64, kernel_size=(3, 3, 3))

    output_layer = Conv3D(filters=n_label, kernel_size=(1, 1, 1), activation='softmax')(u1)

    model = Model(input_layer, output_layer)
    
    return model


def unet_core(x, filter_size=8, kernel_size=(3, 3, 3)):

    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    
    return x
    
def unet_core_dropout(x, filter_size=8, kernel_size=(3, 3, 3)):

    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = SpatialDropout3D(0.1)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    
    return x
    
#### Bayesian U-Net
    
def bayesian_Unet3d(patch_size, n_label, dropout_ratio):

    input_layer = Input(shape=patch_size)
    
    # Encoding part
    d1 = unet_core(input_layer, filter_size = 64, kernel_size=(3, 3, 3))	
    l = Conv3D(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d1)   
    d2 = unet_core(l, filter_size=128, kernel_size=(3, 3, 3))   
    l =  Conv3D(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d2)
    d3 = bayesian_Unet_convBlock(l, filter_size=256, kernel_size=(3, 3, 3), dropout_ratio = dropout_ratio)   
    l = Conv3D(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d3)    
    b = bayesian_Unet_convBlock(l, filter_size=512, kernel_size=(3, 3, 3), dropout_ratio = dropout_ratio)

    
    # Decoding part
    l = Conv3DTranspose(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(b)
    l = concatenate([l, d3], axis=-1)   
    u3 = bayesian_Unet_convBlock(l, filter_size=256, kernel_size=(3, 3, 3), dropout_ratio = dropout_ratio)   
    l = Conv3DTranspose(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u3)
    l = concatenate([l, d2], axis=-1)    
    u2 = unet_core(l, filter_size=128, kernel_size=(3, 3, 3)) 
    l = Conv3DTranspose(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u2)
    l = concatenate([l, d1], axis=-1)
    u1 = unet_core(l, filter_size=64, kernel_size=(3, 3, 3))

	
    output_layer = Conv3D(filters=n_label, kernel_size=(1, 1, 1), activation='softmax')(u1)

    model = Model(input_layer, output_layer)
    
    return model
    
    
def bayesian_Unet_convBlock(x, filter_size, kernel_size, dropout_ratio):

    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = SpatialDropout3D(dropout_ratio)(x, training = True)
    x = BatchNormalization()(x)
    x = PReLU()(x)
#     x = Dropout(dropout_ratio)(x, training = True)
    
    return x

def bayesian_Unet3d_spatial3DConcretDropout(patch_size, n_label):

    input_layer = Input(shape=patch_size)
    
    # Encoding part
    d1 = bayesian_Unet_convBlock_spatial3DConcreteDropout(input_layer, filter_size = 64, kernel_size=(3, 3, 3))
	
    l = Conv3D(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d1)   
    d2 = bayesian_Unet_convBlock_spatial3DConcreteDropout(l, filter_size=128, kernel_size=(3, 3, 3))  
    
    l =  Conv3D(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d2)
    d3 = bayesian_Unet_convBlock_spatial3DConcreteDropout(l, filter_size=256, kernel_size=(3, 3, 3))
    
    l = Conv3D(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d3)    
    b = bayesian_Unet_convBlock_spatial3DConcreteDropout(l, filter_size=512, kernel_size=(3, 3, 3))
    
    # Decoding part
    l = Conv3DTranspose(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(b)
    l = concatenate([l, d3], axis=-1)
    
    u3 = bayesian_Unet_convBlock_spatial3DConcreteDropout(l, filter_size=256, kernel_size=(3, 3, 3))
    
    l = Conv3DTranspose(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u3)
    l = concatenate([l, d2], axis=-1)
    
    u2 = bayesian_Unet_convBlock_spatial3DConcreteDropout(l, filter_size=128, kernel_size=(3, 3, 3))
    
    l = Conv3DTranspose(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u2)
    l = concatenate([l, d1], axis=-1)
    
    u1 = bayesian_Unet_convBlock_spatial3DConcreteDropout(l, filter_size=64, kernel_size=(3, 3, 3))
	
    output_layer = Conv3D(filters=n_label, kernel_size=(1, 1, 1), activation='softmax')(u1)

    model = Model(input_layer, output_layer)
    
    return model

def bayesian_Unet_convBlock_spatial3DConcreteDropout(x, filter_size=8, kernel_size=(3, 3, 3)):
# 	
# 	lengthScale = 0.1;
# 	Thau = 0.01;
# 	N = 24;
# 	weightRegularizer = (lengthScale * lengthScale)/(Thau * N)
# 	dropoutRegullarizer = 1 / (Thau * N)
	
	x = Conv3D(filters=filter_size, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)
	x = Spatial3DConcreteDropout(Conv3D(filters=filter_size, kernel_size=kernel_size, padding='same',kernel_initializer='he_normal'), weight_regularizer = 1e-6, dropout_regularizer = 1e-5)(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)
	
	return x


##### DenseNet

          
def dense_block_aux(x, kernel_size_conv1,kernel_size_conv2, growth_rate):

	x = Conv3D(filters = growth_rate, kernel_size = kernel_size_conv1, padding = 'same', kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)
	x = Conv3D(filters = growth_rate, kernel_size = kernel_size_conv2, padding = 'same', kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)

	#x = Dropout(0.2)(x)		
	return x


def dense_block(x, kernel_size_conv1, kernel_size_conv2, growth_rate, nConv):
	
	for iConv in range(nConv):
		result_dense_block = dense_block_aux(x, kernel_size_conv1, kernel_size_conv2, growth_rate)
		x =  concatenate([x, result_dense_block], axis = -1)
		
	return x


	
def transition_down_block(x, input_shape, kernel_size_conv1, kernel_size_conv2, theta):

	x = Conv3D(filters = int(input_shape * theta), kernel_size = kernel_size_conv1, strides = [1, 1, 1], padding = 'same', kernel_initializer = 'he_normal')(x)
	x = Conv3D(filters = int(input_shape * theta), kernel_size = kernel_size_conv2, strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(x)
	
	return x

def transition_up_block(x, input_shape, kernel_size_conv, theta):
	
	x = Conv3DTranspose(filters = int(input_shape*theta), kernel_size = kernel_size_conv, strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(x)
	
	return x
	

    
def tiramisu(patch_size, n_label):

    input_layer = Input(shape=patch_size)
    growthRate = 14   			
    theta = 1

    # Encoding part
    x = Conv3D(filters = 48, kernel_size = (3, 3, 3), padding = 'same', kernel_initializer = 'he_normal')(input_layer)
  
    d1 = dense_block(x, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 2)
    t1 = transition_down_block(d1, input_shape = d1.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('d1 shape is: ' + str(d1.get_shape().as_list()))
    print('t1 shape is: ' + str(t1.get_shape().as_list()))

    d2 = dense_block(t1, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    t2 = transition_down_block(d2, input_shape = d2.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('d2 shape is: ' + str(d2.get_shape().as_list()))
    print('t2 shape is: ' + str(t2.get_shape().as_list()))
    
    
    d3 = dense_block(t2, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 8)
    t3 = transition_down_block(d3, input_shape = d3.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('d3 shape is: ' + str(d3.get_shape().as_list()))
    print('t3 shape is: ' + str(t3.get_shape().as_list()))
    
    d4 = dense_block(t3, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 12)
    print('d4 shape is: ' + str(d4.get_shape().as_list()))
	
	# Decoding part 
    t1 = transition_up_block(d4, input_shape = d3.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    c1 = concatenate([t1, d3], axis = -1)
    u1 = dense_block(c1, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 8)
    print('t1 up shape is: ' + str(t1.get_shape().as_list()))
    print('u1 up shape is: ' + str(u1.get_shape().as_list()))


    t2 = transition_up_block(u1, input_shape = d2.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    c2 = concatenate([t2, d2], axis = -1)  
    u2 = dense_block(c2, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    print('t2 up shape is: ' + str(t2.get_shape().as_list()))
    print('u2 up shape is: ' + str(u2.get_shape().as_list()))
    
    t3 = transition_up_block(u2, input_shape = d1.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    c3 = concatenate([t3, d1], axis = -1)  
    u3 = dense_block(c3, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 2)
    print('t3 up shape is: ' + str(t3.get_shape().as_list()))
    print('u3 up shape is: ' + str(u3.get_shape().as_list()))
    
        
    output_layer = Conv3D(filters = n_label, kernel_size = (1, 1, 1), activation = 'softmax')(u3)

    
    model = Model(input_layer, output_layer)
    
    return model
    
  
def denseNet(patch_size, n_label):


    input_layer = Input(shape=patch_size)
    growthRate = 16				
    theta = 0.5

    
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
  
    x1 = Conv3D(filters = 32, kernel_size = (3, 3, 3),  strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(x)


    d1 = dense_block(x1, kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (3, 3, 3), growth_rate = growthRate, nConv = 4) 
    t1 = transition_down_block(d1, input_shape = d1.get_shape().as_list()[-1], kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (1, 1, 1), theta = theta)
    print('d1 shape is: ' + str(d1.get_shape().as_list()))
    print('t1 shape is: ' + str(t1.get_shape().as_list()))

    d2 = dense_block(t1, kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    t2 = transition_down_block(d2, input_shape = d2.get_shape().as_list()[-1], kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (1, 1, 1),theta = theta)
    print('d2 shape is: ' + str(d2.get_shape().as_list()))
    print('t2 shape is: ' + str(t2.get_shape().as_list()))
    
    
    d3 = dense_block(t2, kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    t3 = transition_down_block(d3, input_shape = d3.get_shape().as_list()[-1], kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (1, 1, 1), theta = theta)
    print('d3 shape is: ' + str(d3.get_shape().as_list()))
    print('t3 shape is: ' + str(t3.get_shape().as_list()))
    
    d4 = dense_block(t3, kernel_size_conv1 = (1, 1, 1), kernel_size_conv2 = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    
    #Up sampling path
    d1_up = Conv3DTranspose(filters = d1.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(d1) 
    d2_up = Conv3DTranspose(filters = d2.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [4, 4, 4], padding = 'same', kernel_initializer = 'he_normal')(d2) 
    d3_up = Conv3DTranspose(filters = d3.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [8, 8, 8], padding = 'same', kernel_initializer = 'he_normal')(d3)
    d4_up = Conv3DTranspose(filters = d4.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [16, 16, 16], padding = 'same', kernel_initializer = 'he_normal')(d4)
    
    c = concatenate([x, d1_up, d2_up, d3_up, d4_up], axis = -1)
        
    output_layer = Conv3D(filters = n_label, kernel_size = (1, 1, 1), activation = 'softmax')(c)
    
    print('last layer shape is: ' + str(output_layer.get_shape().as_list()))
    
    model = Model(input_layer, output_layer)
    
    return model  
    
