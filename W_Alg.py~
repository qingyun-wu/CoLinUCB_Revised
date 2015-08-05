import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import math
from SPG import *

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



class WknowThetaStruct:
	def __init__(self, featureDimension, eta_, userNum, theta):
		self.theta = theta
		self.userNum = userNum
		self.featureDimension = featureDimension

		self.eta_ = eta_

		self.A = eta_ * np.identity(n = userNum*userNum)
		self.b = np.zeros(userNum*userNum)
		self.W = np.zeros(shape = (userNum, userNum))
		self.CoTheta = np.dot(self.theta, self.W)

		self.f1 = 0.0
		self.f2 = (eta_/2.0)*np.matrix.trace(np.dot(np.transpose(self.W), self.W))
		

		self.BigTheta = np.kron(np.identity(n=userNum), self.theta)
		self.CCA = np.identity(n = featureDimension*userNum)
	def updateParameters(self, articlePicked, click, userID):
		Xi_Matirx = np.zeros(shape = (self.featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		X = vectorize( np.dot(np.transpose(self.theta), Xi_Matirx))

		#self.W = matrixize(np.dot(np.linalg.inv(self.A), self.b), self.userNum)
		#fun = lambda x = np.identity(self.userNum): self.f1 + (1/2.0)*(np.dot( np.dot(articlePicked.featureVector, self.theta)  ,x.T[userID]) - click)**2 + (self.eta_/2.0)*np.matrix.trace(np.dot(np.transpose(x), x))
		def fun(x):
			x = np.asarray(x)
			x = x.reshape(self.userNum, self.userNum)
			return self.f1 + (1/2.0)*(np.dot( np.dot(articlePicked.featureVector, self.theta)  ,x.T[userID]) - click)**2 + (self.eta_/2.0)*np.matrix.trace(np.dot(np.transpose(x), x))

		def fprime(x):
			x = np.asarray(x)
			return np.dot(self.A+ np.outer(X, X), x) -(self.b + click*X) 

		cons = ({'type' : 'ineq',
				'fun' : lambda  x: x },
				{'type' : 'ineq',
				'fun' : lambda x: 1-x},
				{'type': 'eq',
			     'fun' : lambda x : sum(x[0 * self.userNum: 1*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[1 * self.userNum: 2*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[2 * self.userNum: 3 *self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[3 * self.userNum: 4*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[4 * self.userNum: 5*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[5 * self.userNum: 6*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[6 * self.userNum: 7*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[7 * self.userNum: 8*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[8 * self.userNum: 9*self.userNum]) - 1 },
			     {'type': 'eq',
			     'fun' : lambda x : sum(x[9 * self.userNum: 10*self.userNum]) - 1 }
				)
		'''
		res = minimize(fun, self.W, constraints = cons, method ='SLSQP', jac = fprime, options={'disp': False})
		self.W = res.x.reshape(self.userNum, self.userNum)
		#self.W = np.transpose(self.W)
		'''
		

		#print self.W[0]
		#Normalization
		#self.W = normalize(self.W, axis=0, norm='l1')
		self.A += np.outer(X, X)
		self.b += click * X
		self.W = matrixize(np.dot(np.linalg.inv(self.A), self.b), self.userNum)

		self.f1 +=(1/2.0)*(np.dot( np.dot(articlePicked.featureVector, self.theta)  ,self.W.T[userID]))**2
		self.CoTheta = np.dot(self.theta, self.W)
		self.CCA = np.dot(np.dot(self.BigTheta, np.linalg.inv(self.A)), np.transpose(self.BigTheta))

	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)

		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha*var
		#print np.shape(mean)
		return pta




class WStruct:
	def __init__(self, featureDimension, lambda_, eta_, userNum):	
		self.userNum = userNum
		# Basic stat in estimating Theta
		self.T_A = lambda_*np.identity(n = featureDimension*userNum)
		self.T_b = np.zeros(featureDimension*userNum)
		#self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.UserTheta = np.random.random((featureDimension, userNum))
		

		# Basic stat in estimating W
		self.W_A = eta_*np.identity(n = userNum*userNum)
		self.W_b = np.zeros(userNum*userNum)
		#self.W =  np.zeros(shape = (userNum, userNum))
		self.W = np.random.random((userNum, userNum))

		self.CoTheta = np.dot(self.UserTheta, self.W)

		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
		self.CCA = np.identity(n = featureDimension*userNum)

		self.BigTheta = np.kron(np.identity(n=userNum) , self.UserTheta)
		self.W_CCA = np.identity(n = featureDimension*userNum)
		
	def updateParameters(self, articlePicked, click,  userID):	
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.T_A += np.outer(T_X, T_X)	
		self.T_b += click*T_X

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		#print np.shape(W_X)
		self.W_A += np.outer(W_X, W_X)
		self.W_b += click * W_X

		self.UserTheta = matrixize(np.dot(np.linalg.inv(self.T_A), self.T_b), len(articlePicked.featureVector)) 
		self.W = matrixize(np.dot(np.linalg.inv(self.W_A), self.W_b), self.userNum)
		#self.W = normalize(self.W, axis=0, norm='l1')

		#print 'A', self.W_A
		#print 'b', self.W_b
		'''
		plt.pcolor(self.W_b)
		plt.colorbar
		plt.show()
		'''

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , np.linalg.inv(self.T_A)), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)

		self.W_CCA = np.dot(np.dot(self.BigTheta , np.linalg.inv(self.W_A)), np.transpose(self.BigTheta))
	
	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		
		W_var = np.sqrt(np.dot( np.dot(TempFeatureV, self.W_CCA) , TempFeatureV))

		pta = mean + alpha * (var + W_var)
		return pta

		

class WAlgorithm:
	def __init__(self, dimension, alpha, lambda_, eta_, n):  # n is number of users
		self.USERS = WStruct(dimension, lambda_, eta_, n)
		self.dimension = dimension
		self.alpha = alpha

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
	def getW(self, userID):
		#print self.USERS.W
		return self.USERS.W.T[userID]

	def getA(self):
		return self.USERS.A

class WknowThetaAlgorithm(WAlgorithm):	
	def __init__(self, dimension, alpha, lambda_, eta_, n, theta):
		WAlgorithm.__init__(self, dimension, alpha, lambda_, eta_, n)
		self.USERS = WknowThetaStruct(dimension, eta_, n, theta)
		
	def getLearntParameters(self, userID):
		return self.USERS.theta.T[userID]


