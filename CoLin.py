import numpy as np
from util_functions import vectorize, matrixize


class CoLinUCBUserSharedStruct(object):
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.d = featureDimension
		self.A = lambda_*np.identity(n = featureDimension*userNum) # accumulated feature matrix, a dN by dN matrix
		self.CCA = np.identity(n = featureDimension*userNum) # inverse of A, a dN by dN matrix
		self.b = np.zeros(featureDimension*userNum)

		self.AInv = np.linalg.inv(self.A)

		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
		
	def updateParameters(self, articlePicked, click,  userID):
		pass
	
	def getProb(self, alpha, articleFeatureVector, userID):
		TempFeatureV = np.zeros(len(articleFeatureVector)*self.userNum)
		TempFeatureV[float(userID)*self.d:(float(userID)+1)*self.d] = np.asarray(articleFeatureVector)

		mean = np.dot(self.CoTheta.T[userID], articleFeatureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha * var
		return pta


class AsyCoLinUCBUserSharedStruct(CoLinUCBUserSharedStruct):	
	def updateParameters(self, articlePicked_FeatureVector, click,  userID):	
		X = vectorize(np.outer(articlePicked_FeatureVector, self.W.T[userID])) 
		self.A += np.outer(X, X)	
		self.b += click*X

		self.AInv = np.linalg.inv(self.A)

		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked_FeatureVector)) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW, self.AInv), np.transpose(self.BigW))
		

class SyCoLinUCBUserSharedStruct(CoLinUCBUserSharedStruct):
	def __init__(self, featureDimension, lambda_, userNum, W):
		CoLinUCBUserSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = userNum, W= W)
		self.featureVectorMatrix = np.zeros(shape =(featureDimension, userNum))
		self.reward = np.zeros(userNum)
		
	def updateParameters(self, articlePicked_FeatureVector, click, userID):	
		self.featureVectorMatrix.T[userID] = articlePicked_FeatureVector
		self.reward[userID] = click
		
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
		self.AInv = np.linalg.inv(self.A)

		self.UserTheta = matrixize(np.dot(self.AInv, self.b), featureDimension) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))

		
		
#---------------CoLinUCB(fixed user order) algorithms: Asynisized version and Synchorized version		
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
			x_pta = self.USERS.getProb(self.alpha, x.featureVector, userID)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked.featureVector, click, userID)
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoThetaFromCoLinUCB(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return self.USERS.A


class AsyCoLinUCBAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension = dimension, alpha = alpha, lambda_ = alpha, n = n, W = W)
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)
		
		
class syncCoLinUCBAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n, W = W)
		self.USERS = SyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)

	def LateUpdate(self):
		self.USERS.LateUpdate()


#-----------CoLinUCB select user algorithm(only has asynchorize version)-----
class CoLinUCB_SelectUserAlgorithm(AsyCoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		CoLinUCBAlgorithm.__init__(self, dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n, W= W)
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W) 

	def decide(self, pool_articles, AllUsers):
		maxPTA = float('-inf')
		articlePicked = None
		userPicked = None

		for x in pool_articles:
			for user in AllUsers:
				x_pta = self.USERS.getProb(self.alpha, x.featureVector, user.id)
				# pick article with highest Prob
				if maxPTA < x_pta:
					articlePicked = x
					userPicked = user
					maxPTA = x_pta

		return userPicked,articlePicked