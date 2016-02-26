import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime
from util_functions import vectorize, matrixize


class CoLinUserSharedStruct(object):
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.CCA = np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv =  np.linalg.inv(self.A)
		

		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
	def updateParameters(self, articlePicked, click,  userID):
		X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.A += np.outer(X, X)	
		self.b += click*X
		self.AInv =  np.linalg.inv(self.A)

		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.featureVector)) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
	
	def getProb(self, alpha, article, userID):
		
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.featureVector)	
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		
		pta = mean + alpha * var
		
		return pta
	


		
#---------------CoLinUCB(fixed user order) algorithms: Asynisized version and Synchorized version		
class CoLinAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.USERS = CoLinUserSharedStruct(dimension, lambda_, n, W)
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

		self.CanEstimateCoUserPreference = True 
		self.CanEstimateUserPreference = True 
		self.CanEstimateW = False

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

	def getA(self):
		return self.USERS.A



#-----------CoLinUCB select user algorithm(only has asynchorize version)-----
class CoLin_SelectUserAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)  
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

	def decide(self, pool_articles, AllUsers):
		maxPTA = float('-inf')
		articlePicked = None
		userPicked = None

		for x in pool_articles:
			for user in AllUsers:
				x_pta = self.USERS.getProb(self.alpha, x, user.id)
				# pick article with highest Prob
				if maxPTA < x_pta:
					articlePicked = x
					userPicked = user
					maxPTA = x_pta

		return userPicked,articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked, click, userID)
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoThetaFromCoLinUCB(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return self.USERS.A
	
