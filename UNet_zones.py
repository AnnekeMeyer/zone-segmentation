import sys
import numpy as np
import os
import utils
import preprocessing
import evaluation
import SimpleITK as sitk
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout

smooth = 1.

class anisotopic_UNET:

    def get_Tversky(alpha=.3, beta=.7, verb=0):
        def Tversky(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            G_P = alpha * K.sum((1 - y_true_f) * y_pred_f)  # G not P
            P_G = beta * K.sum(y_true_f * (1 - y_pred_f))  # P not G
            return (intersection + smooth) / (intersection + smooth + G_P + P_G)

        def Tversky_loss(y_true, y_pred):
            return -Tversky(y_true, y_pred)

        return Tversky, Tversky_loss


    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def jaccard_distance(self, y_true, y_pred):
        smooth = 100
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    def jaccard_distance(self, y_true, y_pred):
        return -self.jaccard_distance(y_true, y_pred)


    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)

    def weighted_dice_coef_loss(self, y_true, y_pred):
        return -self.afs_weight*self.dice_coef(y_true, y_pred)


    # downsampling, analysis path
    def downLayer(self, inputLayer, filterSize, i, bn=False):

        conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling3D(pool_size=(1, 2, 2))(conv)

        return pool, conv


    # upsampling, synthesis path
    def upLayer(self, inputLayer, concatLayer, filterSize, i, bn=False, do= False):

        up = Conv3DTranspose(filterSize, (2, 2, 2), strides=(1, 2, 2), activation='relu', padding='same',  name='up'+str(i))(inputLayer)
       # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed = 3, name='Dropout_' + str(i))(conv)
        conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)

        return conv



    def get_net(self, nrInputChannels=1, learningRate=5e-5, bn = True, do = False):

        sfs = 16 # start filter size

        inputs = Input((32, 168, 168, nrInputChannels))

        conv1, conv1_b_m = self.downLayer(inputs, sfs, 1, bn)
        conv2, conv2_b_m = self.downLayer(conv1, sfs*2, 2, bn)

        conv3 = Conv3D(sfs*4, (3, 3, 3), activation='relu', padding='same', name='conv' + str(3) + '_1')(conv2)
        if bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(sfs * 8, (3, 3, 3), activation='relu', padding='same',  name='conv' + str(3) + '_2')(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        #conv3, conv3_b_m = downLayer(conv2, sfs*4, 3, bn)

        conv4 = Conv3D(sfs*16 , (3, 3, 3), activation='relu', padding='same',  name='conv4_1')(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)
        if do:
            conv4= Dropout(0.5, seed = 4, name='Dropout_' + str(4))(conv4)
        conv4 = Conv3D(sfs*16 , (3, 3, 3), activation='relu', padding='same',  name='conv4_2')(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)

        #conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
        up1 = Conv3DTranspose(sfs*16, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same', name='up'+str(5))(conv4)
        up1 = concatenate([up1, conv3])
        conv5 = Conv3D(int(sfs*8), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(5)+'_1')(up1)
        if bn:
            conv5 = BatchNormalization()(conv5)
        if do:
            conv5 = Dropout(0.5, seed = 5, name='Dropout_' + str(5))(conv5)
        conv5 = Conv3D(int(sfs*8), (3, 3, 3), activation='relu', padding='same', name='conv'+str(5)+'_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv6 = self.upLayer(conv5, conv2_b_m, sfs*8, 6, bn, do)
        conv7 = self.upLayer(conv6, conv1_b_m, sfs*4, 7, bn, do)



        conv_out = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax')(conv7)

        pz = Lambda(lambda x: x[:, :, :, :, 0], name='pz')(conv_out)
        cz = Lambda(lambda x: x[:, :, :, :, 1], name='cz')(conv_out)
        us = Lambda(lambda x: x[:, :, :, :, 2], name='us')(conv_out)
        afs = Lambda(lambda x: x[:, :, :, :, 3], name='afs')(conv_out)
        bg = Lambda(lambda x: x[:, :, :, :, 4], name='bg')(conv_out)



        model = Model(inputs=[inputs], outputs=[pz, cz, us, afs, bg])
        model.compile(optimizer=Adam(lr=learningRate),
                      loss={'pz': self.dice_coef_loss, 'cz': self.dice_coef_loss, 'us': self.dice_coef_loss,
                            'afs': self.dice_coef_loss, 'bg': self.dice_coef_loss},
                      metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                               'afs': self.dice_coef, 'bg': self.dice_coef})

        return model



def train_model(epochs, learningRate, imgs, gt_list, val_imgs, val_gt_list, foldNr):

    name = 'UNet_zones_Fold' + str(foldNr) + '_LR_' + str(learningRate)

    # keras callbacks
    csv_logger = CSVLogger(name+'.csv', append=True, separator=';')
    model_checkpoint = ModelCheckpoint(name+'.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    earlyStopImprovement = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=1, mode='min')
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=25, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)
    tensorboard = TensorBoard(log_dir='./tensorboard_logs/', write_graph=False, write_grads=False, histogram_freq=0,
                              batch_size=5,
                              write_images=False)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    network = anisotopic_UNET()
    model = network.get_net(learningRate = LR, bn = True, do=True)
    # plot_model(model, to_file='model.png')

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    cb = [csv_logger, model_checkpoint, earlyStopImprovement, LRDecay]

    print('Callbacks: ', cb)

    history = model.fit(imgs, gt_list, batch_size=2, epochs=epochs,
                        verbose=1, validation_data=[val_imgs, val_gt_list], shuffle=True, callbacks=cb)
    model.save(name + '_final.h5')

    return history


def predict(img_arr, modelName):

    network = anisotopic_UNET()
    model = network.get_net(bn=True, do=False)
    model.load_weights(modelName)
    prediction = model.predict([img_arr], batch_size=2, verbose=1)

    #np.save(os.path.join(out_dir, 'predicted_'+ modelName[:-3] + '.npy'), out)

    return prediction


if __name__ == '__main__':


    # input directory that contains three orthogonal images (tra, sag, cor), which are needed for preprocessing
    inputDir = 'data-test/ProstateX-0217'
    arr = preprocessing.preprocessImage(inputDir)
    pred_arr = predict(arr, modelName = 'model/model.h5')
    pred_arr = np.asarray(pred_arr)

    roi_tra = sitk.ReadImage(os.path.join(inputDir, 'roi_tra.nrrd'))

    # removes isolated segments from prediction
    pred_arr = utils.removeIslands(pred_arr[:,0,:,:,:])
    # convert to SimpleITK image. Zone affiliation is marked by intensity value (pz=1, cz=2, us=3, afs=4)
    pred_img = utils.convertArrayToMuliLabelImage(pred_arr, templateImg = roi_tra)

    sitk.WriteImage(pred_img, os.path.join(inputDir, 'predicted_roi.nrrd'))

