#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Machine Learning
# Date: 9/21/18
# Author: bryce
import numpy as np

if __name__ == "__main__":
	Matrix_A = np.loadtxt("data/matrixA.txt",delimiter=",",dtype=int)
	Matrix_B = np.loadtxt("data/matrixB.txt",delimiter=",",dtype=int)
	A = np.array(Matrix_A)
	B = np.array(Matrix_B)
	ans = np.dot(A,B)
	ans_one = np.sort(ans)
	#print (ans_one)
	np.savetxt('ans_one.txt',ans_one,fmt="%d")
