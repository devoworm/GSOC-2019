from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import cv2

augumented_images_path = './Aug_data/Train/'
original_images_path = './Data/train/'

original_images_list = os.listdir(original_images_path)


h = 620
w = 388

def remove_border(img, img_name):
	cropped_image = img[30:650, 60:]
	cropped_image = Image.fromarray(cropped_image)
	cropped_image.save( original_images_path + img_name)

def flip_left_right(img, img_name): #Flipping an image left to right
	im = np.fliplr(img)
	im = Image.fromarray(im)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_lr.'+img_name_list[1])

def flip_up_down(img, img_name): #Flipping an image right to left
	im = np.flipud(img)
	im = Image.fromarray(im)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_ud.'+img_name_list[1])

def rotate_image(img, img_name): #Rotate an image at different angels with a difference of 20 degrees
	for i in range(20,360, 20):
		imgR = ndimage.rotate(img, i, reshape=False)
		im = Image.fromarray(imgR)
		img_name_list = img_name.split('.')
		im.save(augumented_images_path + img_name_list[0]+'_'+str(i)+'.'+img_name_list[1])

def shift_image(img, img_name):
	M = np.float32([[1,0,50],[0,1,0]]) #Shift left by 50 pixels
	dst = cv2.warpAffine(img,M,(w,h))
	im = Image.fromarray(dst)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_sl.'+img_name_list[1])

	M = np.float32([[1,0,-50],[0,1,0]]) #Shift right by 50 pixels
	dst = cv2.warpAffine(img,M,(w,h))
	im = Image.fromarray(dst)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_sr.'+img_name_list[1])

	M = np.float32([[1,0,0],[0,1,50]]) #Shift down by 50 pixels
	dst = cv2.warpAffine(img,M,(w,h))
	im = Image.fromarray(dst)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_sd.'+img_name_list[1])

	M = np.float32([[1,0,0],[0,1,-50]]) #Shift up by 50 pixels
	dst = cv2.warpAffine(img,M,(w,h))
	im = Image.fromarray(dst)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_su.'+img_name_list[1])

def histogram_eq(img, img_name): #Histogram Equalization
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
	equ = clahe.apply(img_gray)
	im = Image.fromarray(equ)
	img_name_list = img_name.split('.')
	im.save(augumented_images_path + img_name_list[0]+'_he.'+img_name_list[1])


for i in range(len(original_images_list)):
	img_name = original_images_list[i]
	img = Image.open(original_images_path+img_name)
	img = np.array(img)
	# remove_border(img, img_name) #only used once
	print("For image", original_images_list[i])
	print('*'*40)
	flip_up_down(img, original_images_list[i])
	print("image flipped up down")

	flip_left_right(img, original_images_list[i])
	print("image flipped left right")

	rotate_image(img, original_images_list[i])
	print("image rotated in all direction")

	shift_image(img, original_images_list[i])
	print("image shifted in all direction")


	histogram_eq(img, original_images_list[i])
	print("image histogram equalization done")
	print('\n')

# 8232 images generated from 343 images