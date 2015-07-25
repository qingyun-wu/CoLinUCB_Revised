import numpy as np
from scipy.linalg import sqrtm
import math
from Algori import *



class LinUCB_SelectUserAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ )) 

		self.dimension = dimension
		self.alpha = alpha

	def decide(self, pool_articles, AllUsers):
		maxPTA = float('-inf')
		articlePicked = None
		userPicked = None
		AllUsers = list(np.random.permutation(AllUsers)) 
		for x in pool_articles:
			for user in AllUsers:
				x_pta = self.users[user.id].getProb(self.alpha, self.users, x)
				# pick article with highest Prob
				if maxPTA < x_pta:
					articlePicked = x
					userPicked = user
					maxPTA = x_pta

		return userPicked, articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta


class CoSingleAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):
		self.users = []
		for i in range(n):
			self.users.append(CoSingleStruct(dimension, lambda_, n, W, i))
		self.dimension = dimension
		self.alpha = alpha
		self.W = W
	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, x, userID)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click, userID)
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta.T[userID]
	def getCoThetaFromCoLinUCB(self,userID):
		return self.users[userID].CoTheta.T[userID]


class CoLinUCB_SelectUserAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n, W)
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


class AsyCoLinUCB_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)
		
class syncCoLinUCB_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = SyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)

	def LateUpdate(self):
		self.USERS.LateUpdate()



class GOBLin_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
	def getLearntParameters(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]

