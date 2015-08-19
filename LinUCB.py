import numpy as np

class LinUCBUserStruct:
	def __init__(self, featureDimension, lambda_):
		self.d = featureDimension
		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.zeros(self.d)

	def updateParameters(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		
	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, article_FeatureVector):
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		pta = mean + alpha * var
		return pta

class Uniform_LinUCBAlgorithm(object):
	def __init__(self, dimension, alpha, lambda_):
		self.dimension = dimension
		self.alpha = alpha
		self.USER = LinUCBUserStruct(dimension, lambda_)

		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = True 
		self.CanEstimateW = False
	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.USER.getProb(self.alpha, x.featureVector)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.USER.updateParameters(articlePicked.featureVector, click)
	def getCoTheta(self, userID):
		return self.USER.UserTheta



#---------------LinUCB(fixed user order) algorithm---------------
class N_LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, lambda_ )) 

		self.dimension = dimension
		self.alpha = alpha

		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = True 
		self.CanEstimateW = False
	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, x.featureVector)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.featureVector, click)
		
	def getCoTheta(self, userID):
		return self.users[userID].UserTheta


#-----------LinUCB select user algorithm-----------
class LinUCB_SelectUserAlgorithm(N_LinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n):  # n is number of users
		N_LinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n)

	def decide(self, pool_articles, AllUsers):
		maxPTA = float('-inf')
		articlePicked = None
		userPicked = None
		
		for x in pool_articles:
			for user in AllUsers:
				x_pta = self.users[user.id].getProb(self.alpha, x.featureVector)
				# pick article with highest Prob
				if maxPTA < x_pta:
					articlePicked = x
					userPicked = user
					maxPTA = x_pta

		return userPicked, articlePicked