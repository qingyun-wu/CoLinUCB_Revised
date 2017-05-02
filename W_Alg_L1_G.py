import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import math
from util_functions import *
import time
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp

class WStruct_l1_G:
	def __init__(self, featureDimension, lambda_,  userNum, W, windowSize, RankoneInverse):
		self.windowSize = windowSize
		self.counter = 0
		self.RankoneInverse = RankoneInverse
		self.userNum = userNum
		self.lambda_ = lambda_
		self.alpha_t = 0.0
		self.delta = 1000000
		self.TrueW = W
		# Basic stat in estimating Theta
		self.lambda_I = lambda_*np.identity(n = featureDimension*userNum)
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		#self.UserTheta = np.random.random((featureDimension, userNum))
		self.AInv = np.linalg.inv(self.A)
		
		#self.W = np.random.random((userNum, userNum))
		self.W = np.identity(n = userNum)
		#self.W = self.TrueW
		self.Wlong = vectorize(self.W)
		self.batchGradient = np.zeros(userNum*userNum)

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
		self.CCA = np.identity(n = featureDimension*userNum)
		self.BigTheta = np.kron(np.identity(n=userNum) , self.UserTheta)
		self.W_X_arr = []
		self.W_y_arr = []
		for i in range(userNum):
			self.W_X_arr.append([])
			self.W_y_arr.append([])
		
	def updateParameters(self, articlePicked, click,  userID):	
		self.counter +=1
		self.Wlong = vectorize(self.W)
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.A += np.outer(T_X, T_X)	
		self.b += click*T_X
		if self.RankoneInverse:
			temp = np.dot(self.AInv, T_X)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(T_X),temp))
		else:
			self.AInv =  np.linalg.inv(self.A)
		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.featureVector)) 

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		W_X_current = np.dot(np.transpose(self.UserTheta), articlePicked.featureVector)

		self.W_X_arr[userID].append(W_X_current)
		self.W_y_arr[userID].append(click)

		
		#print self.windowSize
		if self.counter%self.windowSize ==0:
			for i in range(len(self.W)):
				if len(self.W_X_arr[i]) !=0:
					def fun(w):
						w = np.asarray(w)
						return np.sum((np.dot(self.W_X_arr[i], w) - self.W_y_arr[i])**2, axis = 0) + self.lambda_*np.linalg.norm(w,1)
						#return np.sum((np.dot(self.W_X_arr[i], w) - self.W_y_arr[i])**2, axis = 0) + self.lambda_*np.linalg.norm(w-self.TrueW,2)

					def evaluateGradient(w):
						w = np.asarray(w)
						X = np.asarray(self.W_X_arr[i])
						y = np.asarray(self.W_y_arr[i])
						grad = np.dot(np.transpose(X) , ( np.dot(X,w)- y)) + self.lambda_ * np.sign(w)
						return grad

					current = self.W.T[i]
					res = minimize(fun, current, constraints = getcons(len(self.W)), method = 'SLSQP', jac = evaluateGradient, bounds=getbounds(len(self.W)), options={'disp': False})
					self.W.T[i] = res.x
					#print self.W.T[i], sum(self.W.T[i])
			
			if self.windowSize< 2000:
				self.windowSize = self.windowSize*2 
			
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)
	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha * var
		#pta = mean + alpha * var
		return pta



		
class LearnWAlgorithm_l1_G:
	def __init__(self, dimension, alpha, lambda_, n, W,  windowSize, RankoneInverse = False):  # n is number of users
		self.USERS = WStruct_l1_G(dimension, lambda_, n, W, windowSize, RankoneInverse)
		self.dimension = dimension
		self.alpha = alpha

		self.CanEstimateUserPreference = True
		self.CanEstimateCoUserPreference =  True
		self.CanEstimateW = True

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.USERS.getProb(self.alpha, x, userID)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked, click, userID)
		
	def getTheta(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]
	def getW(self, userID):
		#print self.USERS.W
		return self.USERS.W.T[userID]

	def getA(self):
		return self.USERS.A
	def showLearntWheatmap(self):
		showheatmap(self.USERS.W)
		#return self.USERS.W.T
	def getWholeW(self):
		return self.USERS.W

