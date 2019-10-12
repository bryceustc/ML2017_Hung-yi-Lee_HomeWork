#!/usr/local/bin/python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
	Matrix_A = np.loadtxt("data/matrixA.txt",delimiter=",",dtype=int)
	Matrix_B = np.loadtxt("data/matrixB.txt",delimiter=",",dtype=int)
	#loadtxt("地址"，delimiter=""分隔符，dtype= 数据类型 默认float)
	A = np.array(Matrix_A)
	B = np.array(Matrix_B)
	ans = np.dot(A,B)
	ans_one = np.sort(ans)
	#np.sort(a,axis,kind,order) 
	#a:排序数组，axis:沿着它排序数组的轴，如果没有数组会被展开，axis=0 按列排序，axis= 1 按行排序,默认是按行排序
	#kind: 默认quicksort(快速排序)，还有mergesort(归并排序)，heapsort(堆排序)
	#orde: 如果数组包含字段，则是要排序的字段
	print (ans_one)
	np.savetxt('ans_one.txt',ans_one,fmt="%d")
