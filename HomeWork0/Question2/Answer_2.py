#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Machine Learning
# Date: 9/22/18
# Author: bryce
from PIL import Image
from PIL import ImageChops
from PIL import ImageDraw

if __name__ == '__main__':
	img = Image.open("data/lena.png")
	img_modified = Image.open("data/lena_modified.png")
	x,y = img_modified.size
	for i in range(0,x):
		for j in range(0,y):
			if img.getpixel((i,j)) == img_modified.getpixel((i,j)):
				img_modified.putpixel((i,j),255)
	img_modified.show()
	img_modified.save('ans_two.png')
