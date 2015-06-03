import numpy as np
from scipy.linalg import sqrtm
import math

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

class LinUCBUserStruct:
	def __init__(self, featureDimension, userID, lambda_):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		self.b = np.zeros(featureDimension)
		self.UserTheta = np.zeros(featureDimension)

	def updateParameters(self, articlePicked, click):
		featureVector = articlePicked.featureVector
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)
		
	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, users, article):
		featureVector = article.featureVector
		mean = np.dot(self.getTheta(), featureVector)
		var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(self.getA())), featureVector))
		pta = mean + alpha * var
		return pta

class GOBLinSharedStruct:
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)

		self.theta = np.dot(np.linalg.inv(self.A), self.b)
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=featureDimension))) )
		self.STBigW = sqrtm(np.kron(W, np.identity(n=featureDimension)))
	def updateParameters(self, articlePicked, click, userID):
		featureVectorM = np.zeros(shape =(len(articlePicked.featureVector), self.userNum))
		featureVectorM.T[userID] = articlePicked.featureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		self.A += np.outer(CoFeaV, CoFeaV)
		self.b += click * CoFeaV

		self.theta = np.dot(np.linalg.inv(self.A), self.b)
	def getProb(self,alpha , article, userID):
		featureVectorM = np.zeros(shape =(len(article.featureVector), self.userNum))
		featureVectorM.T[userID] = article.featureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)

		mean = np.dot(np.transpose(self.theta), CoFeaV)
	
		var = np.sqrt( np.dot( np.dot(CoFeaV, np.linalg.inv(self.A)) , CoFeaV))
		pta = mean + alpha * var
		return pta

class CoLinUCBUserSharedStruct(object):
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.CCA = np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)

		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
	def updateParameters(self, articlePicked, click,  userID):
		pass
	
	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)

		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha * var
		return pta

class AsyCoLinUCBUserSharedStruct(CoLinUCBUserSharedStruct):	
	def updateParameters(self, articlePicked, click,  userID):	
		X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.A += np.outer(X, X)	
		self.b += click*X

		self.UserTheta = matrixize(np.dot(np.linalg.inv(self.A), self.b), len(articlePicked.featureVector)) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , np.linalg.inv(self.A)), np.transpose(self.BigW))
		

class SyCoLinUCBUserSharedStruct(CoLinUCBUserSharedStruct):
	def __init__(self, featureDimension, lambda_, userNum, W):
		CoLinUCBUserSharedStruct.__init__(self, featureDimension, lambda_, userNum, W)
		self.featureVectorMatrix = np.zeros(shape =(featureDimension, userNum))
		self.reward = np.zeros(userNum)
	def updateParameters(self, articlePicked, click, userID):	
		self.featureVectorMatrix.T[userID] = articlePicked.featureVector
		self.reward[userID] = click
		featureDimension = len(self.featureVectorMatrix.T[userID])
		
	def LateUpdate(self):
		featureDimension = self.featureVectorMatrix.shape[0]
		current_A = np.zeros(shape = (featureDimension* self.userNum, featureDimension*self.userNum))
		current_b = np.zeros(featureDimension*self.userNum)		
		for i in range(self.userNum):
			X = vectorize(np.outer(self.featureVectorMatrix.T[i], self.W.T[i])) 
			XS = np.outer(X, X)	
			current_A += XS
			current_b += self.reward[i] * X
		self.A += current_A
		self.b += current_b

		self.UserTheta = matrixize(np.dot(np.linalg.inv(self.A), self.b), featureDimension) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , np.linalg.inv(self.A)), np.transpose(self.BigW))
		
		
class LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ )) 

		self.dimension = dimension
		self.alpha = alpha

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users, x)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta


class CoLinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n, W)
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

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
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoThetaFromCoLinUCB(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return self.USERS.A


class AsyCoLinUCBAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)
		
class syncCoLinUCBAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = SyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)

	def LateUpdate(self):
		self.USERS.LateUpdate()


class GOBLinAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
	def getLearntParameters(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]

