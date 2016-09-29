import numpy as np
from LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
class COFIBAUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension, lambda_, userID):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension, userID = userID,lambda_= lambda_)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.d = featureDimension
	def updateParameters(self, articlePicked_FeatureVector, click,alpha_2):

		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)

	def updateParametersofClusters(self,clusters,userID,Graph,users):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		for i in range(len(clusters)):
			if clusters[i] == clusters[userID]:
				self.CA += (users[i].A - self.I)
				self.Cb += users[i].b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector,time):
		mean = np.dot(self.CTheta, article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
		return pta

class COFIBAAlgorithm(LinUCBAlgorithm):
	def __init__(self,dimension,alpha,alpha_2,lambda_, n,itemNum,cluster_init = 'Erdos-Renyi'):  
		self.time = 0
		self.dimension = dimension
		self.alpha = alpha
		self.alpha_2 = alpha_2
		self.cluster_init= cluster_init
		self.itemNum = itemNum

		#Every user host an algorithm which operates a linear bandit algorithm
		self.users = []
		for i in range(n):
			self.users.append(COFIBAUserStruct(dimension,lambda_, i)) 

		#Init a single cluster over item set
		if self.cluster_init == 'Erdos-Renyi':
			p = 3 * math.log(itemNum)/itemNum
			self.IGraph = np.random.choice([0, 1], size=(self.itemNum,self.itemNum), p=[1-p, p])
		else:
			self.IGraph = np.ones([self.itemNum, self.itemNum])
		self.Iclusters = []
		N_components_Item, components_Item = connected_components(csr_matrix(self.IGraph))
		self.Iclusters = components_Item
		self.N_components_Item = N_components_Item


		#Init a family of clusters over users set
		self.UGraph = []
		self.Uclusters = []
		for i in range(self.N_components_Item):
			if self.cluster_init == 'Erdos-Renyi':
				p = 3 * math.log(n)/n
				self.UGraph.append(np.random.choice([0, 1], size=(n,n), p=[1-p, p]))
			else:
				self.UGraph.append(np.ones([n,n])) 
			self.Uclusters.append([])
			N_components_U, components_U = connected_components(csr_matrix(self.UGraph[i]))
			self.Uclusters[i] = components_U
		self.UserNeighbor = {}

		self.CanEstimateCoUserPreference = False
		self.CanEstimateUserPreference = False
		self.CanEstimateW = False
			
	def decide(self,pool_articles,userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			itemClusterNum = self.Iclusters[x.id]   #Get the cluster number of item
			self.updateUserClusters(userID, x.featureVector, itemClusterNum)   # get the user clustering based on item x
			self.users[userID].updateParametersofClusters(self.Uclusters[itemClusterNum],userID,self.UGraph, self.users)
			x_pta = self.users[userID].getProb(self.alpha, x.featureVector,self.time)
			#print 'itemClusterNum', itemClusterNum
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x.id
				featureVectorPicked = x.featureVector
				picked = x
				maxPTA = x_pta
		self.time +=1

		return picked
	def updateParameters(self, featureVector, click,userID):
		self.users[userID].updateParameters(featureVector, click, self.alpha_2)
	def updateUserClusters(self,userID, articlePicked_FeatureVector, itemClusterNum):
		n = len(self.users)
		for j in range(n):
			diff = math.fabs(np.dot( self.users[userID].UserTheta, articlePicked_FeatureVector )- np.dot( self.users[j].UserTheta, articlePicked_FeatureVector))
			CB = self.alpha_2* (np.sqrt(np.dot(np.dot(articlePicked_FeatureVector, self.users[userID].AInv),  articlePicked_FeatureVector)) + np.sqrt(np.dot(np.dot(articlePicked_FeatureVector, self.users[j].AInv),  articlePicked_FeatureVector))) * np.sqrt(np.log10(self.time+1))
			#print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
			if diff > CB:
				self.UGraph[itemClusterNum][userID][j] = 0
				self.UGraph[itemClusterNum][j][userID] = self.UGraph[itemClusterNum][userID][j]
		N_components, component_list = connected_components(csr_matrix(self.UGraph[itemClusterNum]))
		#print 'N_components:',N_components
		self.Uclusters[itemClusterNum] = component_list
		return N_components
	def updateItemClusters(self, userID, chosenItem, itemClusterNum, articlePool):
		m = self.itemNum
		n = len(self.users)
		#UserNeighbor = {}
		for a in articlePool:
			if self.IGraph[chosenItem.id][a.id] == 1:
				#UserNeighbor[a.id] = np.ones([n,n])
				for i in range(n):
					diff = math.fabs(np.dot( self.users[userID].UserTheta, a.featureVector )- np.dot( self.users[i].UserTheta, a.featureVector))
					CB = self.alpha_2* (np.sqrt(np.dot(np.dot(a.featureVector, self.users[userID].AInv),  a.featureVector)) + np.sqrt(np.dot(np.dot(a.featureVector, self.users[i].AInv),  a.featureVector))) * np.sqrt(np.log10(self.time+1))
					if diff > CB:
						self.UserNeighbor[a.id][userID][i] = 0
						self.UserNeighbor[a.id][i][userID] = 0
				if not np.array_equal(UserNeighbor[a.id], self.UGraph[itemClusterNum]):
					self.IGraph[chosenItem.id][a.id] = 0
					self.IGraph[a.id][chosenItem.id] = 0
					#print 'delete edge'
		self.N_components_Item, component_list_Item = connected_components(csr_matrix(self.IGraph))
		self.Iclusters = component_list_Item

		# For each new item cluster, allocate a new connected graph over users representing a single user clsuter 
		self.UGraph = []
		self.Uclusters = []
		for i in range(self.N_components_Item):
			if self.cluster_init =='Erdos-Renyi':
				p = 3 * math.log(len(self.users))/len(self.users)
				self.UGraph.append(np.random.choice([0, 1], size=(len(self.users),len(self.users)), p=[1-p, p]))
			else:
				self.UGraph.append(np.ones([len(self.users), len(self.users)]) ) 
			self.Uclusters.append([])
			N_components_U, components_U = connected_components(csr_matrix(self.UGraph[i]))
			self.Uclusters[i] = components_U
		return self.N_components_Item
			
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta
	






