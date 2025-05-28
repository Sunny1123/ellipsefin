import pytest
import numpy as np
from denoise import lm

def test_lm():
	x = np.arange(10)
	y = 5*x +10
	x = np.array([np.repeat(1,10),x]).T
	w = np.repeat(1,10)
	expected = np.array([10.,5.])
	res = lm.lfit(y,x,w)
	assert np.allclose(expected,res)

def test_llk():
	y=np.arange(25).reshape(5,5)
	print(y)
	w=[[0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],
		[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
		[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
	pos = np.array([2,2])
	expected = np.array([12,5,1])
	result = lm.LocalKfit(y,pos,w,1)
	assert np.allclose(expected,result)

def test_lqk():
	y = np.array([[2,1,2],[1,1,3],[2,3,6]])
	w = [[0,0,0,1,1,1,2,2,2],
		[0,1,2,0,1,2,0,1,2],
		[1,1,1,1,1,1,1,1,1]]
	pos =np.array([1,1])
	expected = np.repeat(1,6)
	result = lm.LocalKfit(y,pos,w,2)
	assert np.allclose(expected,result.T)

def test_Kernel():
	expected = [[0,1,1,1,2],
				[1,0,1,2,1],
				[1,1,1,1,1]]
	a=lm.Kernel(3,3,np.array([1,1]))
	b=lm.Kernel(3,3,[1,1])
	result1 = a.eval([1,1])
	result2 = b.eval([1,1])
	assert np.allclose(expected,result1)
	assert np.allclose(expected,result2)

def test_llkwkernnel():
	y = np.arange(9).reshape([3,3])
	pos =np.array([1,1])
	w = lm.Kernel(3,3,[1,1]).eval(pos)
	expected = [4,3,1]
	result = lm.LocalKfit(y,pos,w,1)
	assert np.allclose(expected,result)

def test_lqkwkernnel():
	y = np.arange(9).reshape([3,3])
	pos =np.array([1,1])
	w = lm.Kernel(3,3,[1,1]).eval(pos)
	expected = [4,3,1,0,0,0]
	result = lm.LocalKfit(y,pos,w,2)
	assert np.allclose(expected,result)