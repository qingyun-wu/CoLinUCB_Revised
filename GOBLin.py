import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime

from util_functions import vectorize, matrixize
from CoLin import CoLinAlgorithm, CoLin_SelectUserAlgorithm




class GOBLinSharedStruct:
	def __init__(self, featureDimension, lambda_, userNum, W, RankoneInverse):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv , self.b)
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=featureDimension))) )
		self.STBigW = sqrtm(np.kron(W, np.identity(n=featureDimension)))
		self.RankoneInverse = RankoneInverse
	def updateParameters(self, articlePicked, click, userID):
		featureVectorM = np.zeros(shape =(len(articlePicked.featureVector), self.userNum))
		featureVectorM.T[userID] = articlePicked.featureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		self.A = self.A + np.outer(CoFeaV, CoFeaV)
		self.b = self.b + click * CoFeaV

		if self.RankoneInverse:
			temp = np.dot(self.AInv, CoFeaV)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(CoFeaV),temp))
		else:
			self.AInv =  np.linalg.inv(self.A)
	
		self.theta = np.dot(self.AInv, self.b)
	def getProb(self,alpha , article, userID):
		
		featureVectorM = np.zeros(shape =(len(article.featureVector), self.userNum))
		featureVectorM.T[userID] = article.featureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		
		mean = np.dot(np.transpose(self.theta), CoFeaV)		
		a = np.dot(CoFeaV, self.AInv)
		var = np.sqrt( np.dot( np.dot(CoFeaV, self.AInv) , CoFeaV))
		
		pta = mean + alpha * var
		
		return pta
# inherite from CoLinUCBAlgorithm
class GOBLinAlgorithm(CoLinAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W, RankoneInverse = False):
		CoLinAlgorithm.__init__(self, dimension, alpha, lambda_, n, W, RankoneInverse)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W, RankoneInverse)

		self.CanEstimateCoUserPreference = True 
		self.CanEstimateUserPreference = False
		self.CanEstimateW = False
	def getCoTheta(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]

#inherite from CoLinUCB_SelectUserAlgorithm
class GOBLin_SelectUserAlgorithm(CoLin_SelectUserAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W, RankoneInverse = False):
		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension, alpha, lambda_, n, W, RankoneInverse)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
	def getCoTheta(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]