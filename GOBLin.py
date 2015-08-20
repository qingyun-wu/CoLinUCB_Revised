import numpy as np
from scipy.linalg import sqrtm

from util_functions import vectorize, matrixize
from CoLin import CoLinUCBAlgorithm, CoLinUCB_SelectUserAlgorithm

class GOBLinSharedStruct:
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.d = featureDimension
		self.A = lambda_*np.identity(n = self.d*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv, self.b)
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=self.d))) )
		self.STBigW = sqrtm(np.kron(W, np.identity(n=self.d)))
	
	def updateParameters(self, articlePicked_FeatureVector, click, userID):
		featureVectorV = np.zeros(self.d*self.userNum)
		featureVectorV[float(userID)*self.d:(float(userID)+1)*self.d] = np.asarray(articlePicked_FeatureVector)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		self.A += np.outer(CoFeaV, CoFeaV)
		self.b += click * CoFeaV
		self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv, self.b)
	
	def getProb(self,alpha , article_FeatureVector, userID):
		featureVectorV = np.zeros(self.d*self.userNum)
		featureVectorV[float(userID)*self.d:(float(userID)+1)*self.d] = np.asarray(article_FeatureVector)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)

		mean = np.dot(np.transpose(self.theta), CoFeaV)
		var = np.sqrt( np.dot( np.dot(CoFeaV, self.AInv) , CoFeaV))
		pta = mean + alpha * var
		return pta
	
# inherite from CoLinUCBAlgorithm
class GOBLinAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension = dimension, alpha = alpha, lambda_ = lambda_, n=n, W= W)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)

		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = True
		self.CanEstimateW = False
		
	def getCoTheta(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]

#inherite from CoLinUCB_SelectUserAlgorithm
class GOBLin_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n, W = W)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = True 
		self.CanEstimateW = False
	
	def getCoTheta(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]