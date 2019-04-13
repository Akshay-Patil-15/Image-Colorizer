import keras
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import cv2
import PIL
import matplotlib.pyplot as plt
import PySimpleGUI as sg 
import numpy as np

imsize = 256

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

t = True

model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')
model.load_weights("newfacemodel.h5")
model.compile(optimizer='rmsprop', loss='mse')
print("Weights Loaded")


def Color(bwimage, destination):
    color_me = []
    arr = []
    narr = []
    
    for filename in os.listdir(bwimage):
        arr = img_to_array(load_img(bwimage+filename))
        narr = cv2.resize(arr, (imsize,imsize))
        color_me.append(narr)
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))
    
    output = model.predict(color_me)
    output = output * 128

    i = 0
    dn = destination

    for i in range(len(output)):
        cur = np.zeros((imsize,imsize, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave(dn + "img_"+str(i)+".jpg", lab2rgb(cur))


def Resize(pre):
    i=0
    for filename in os.listdir(pre):
        arr = cv2.imread(pre + filename)
        narr = cv2.resize(arr, (imsize,imsize))
        cv2.imwrite(pre + "img_"+str(i)+".jpg", narr)
        i = i + 1


def ToBW(pre, destination):
    i=0
    for filename in os.listdir(pre):
        arr = cv2.imread(pre + filename)
        narr = cv2.resize(arr, (imsize,imsize))
        narr = cv2.cvtColor(narr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(destination + "img_"+str(i)+".jpg", narr)
        i = i + 1

def Accuracy(og, destination):
	i = 0
	dn = destination + 'img_'
	for image in os.listdir(og):
	    col = img_to_array(load_img(og+image))
	    cnn = img_to_array(load_img(dn + str(i) + '.jpg'))
	    ls = abs(col - cnn)
	    ls = np.array(ls, dtype='float')
	    ls = ls / 255
	    plt.imshow(ls)
	    plt.show()
	    print("Max percent pixel error :", np.max(ls))
	    print("Mean percent pixel error :", np.mean(ls))
	    print("Median percent pixel error :", np.median(ls))
	    print("\n")
	    i = i + 1

def Enhance(destination):
	i = 0
	k = 0
	dn = destination + 'img_'

	if not os.path.exists(destination + 'Enhanced'):
		os.mkdir(destination + 'Enhanced')

	for image in os.listdir(destination):
		if os.path.exists(destination + 'Enhanced') and k == 0:
			k = 1
			pass
		else:	
			img = PIL.Image.open(dn + str(i) + '.jpg')
			con = PIL.ImageEnhance.Color(img)
			img2 = con.enhance(1.5)
			PIL.ImageFile.MAXBLOCK = 2**20
			img2.save(destination + 'Enhanced/img_' + str(i) + '.jpg', "JPEG", quality=100, optimize=True, progressive=True)
			i = i + 1


while(t):
	event, (bwimage, destination, pre, og) = sg.Window('Image Colorizer').Layout([ [sg.Text('B/W Image Folder Path')], 
	                                                                   [sg.Input(), sg.FolderBrowse()],
	                                                                   [sg.Text('Destination Folder Path')], 
	                                                                   [sg.Input(), sg.FolderBrowse()], 
	                                                                   [sg.Text('Preprocessing Images Folder Path')], 
	                                                                   [sg.Input(), sg.FolderBrowse()],
	                                                                   [sg.Text('Original Images Folder Path')], 
	                                                                   [sg.Input(), sg.FolderBrowse()], 
	                                                                   [sg.OK(), sg.Cancel(), sg.Button('Resize'), sg.Button('B/W'), sg.Button('Accuracy'), sg.Button('Enhance')] ]).Read() 
	   
	
	if event == 'Cancel' or event == None:
		t = False
		break 

	bwimage = bwimage + "/"
	destination = destination + "/"
	pre = pre + "/"
	og = og + "/"
	  
	if event == 'OK' and bwimage != '' and destination != '':
	  	Color(bwimage, destination)
	  	sg.Popup('Image colorized in destination set by you!')
	   
	if event == 'OK' and bwimage == '' or destination == '':
	    sg.Popup('Select valid paths!')
	   
	if event == 'Resize' and pre !='':
	    Resize(pre)
	    sg.Popup('Resizing done!')

	if event == 'B/W' and pre !='' and bwimage != '':
	    ToBW(pre, destination)
	    sg.Popup('Conversion to B/W done!')

	if event == 'Accuracy' and og != '' and destination !='':
		Accuracy(og, destination)

	if event == 'Enhance' and destination != '':
		Enhance(destination)
		sg.Popup('Results are Enhanced!')