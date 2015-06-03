import numpy as np
from operator import itemgetter      #for easiness in sorting and finding max and stuff
from matplotlib.pylab import *
from random import sample
from scipy.sparse import csgraph 
import os
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, result_folder, save_address
from util_functions import *
from Articles import *
from Users import *
#from Algori import *
from Algori import *
from eGreedyUCB1 import *
from scipy.linalg import sqrtm
import math

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, users, 
					batchSize = 1000,
					noise = lambda : 0,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					NoiseScale = 0,
					epsilon = 1, Gepsilon = 1):

		self.simulation_signature = signature
		self.type = type_

		self.dimension = dimension
		self.iterations = iterations
		self.noise = noise
		self.articles = articles 
		self.users = users

		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		self.W = self.initializeW(epsilon)
		self.GW = self.initializeGW(Gepsilon)
		self.NoiseScale = NoiseScale
	
	# create user connectivity graph
	def initializeW(self, epsilon):
		n = len(self.users)	
		'''
		W = np.zeros(shape = (n, n))
		for ui in self.users:
			sSim = 0
			for uj in self.users:
				sim = np.dot(ui.theta, uj.theta)
 				if ui.id == uj.id:
 					sim *= 1.0
				W[ui.id][uj.id] = sim
				sSim += sim
				
			W[ui.id] /= sSim
			
			for i in range(n):
				print '%.3f' % W[ui.id][i],
			print ''
		'''

		#random generation
 		a = np.ones(n-1) 
 		b =np.ones(n);
 		c = np.ones(n-1)
 		k1 = -1
 		k2 = 0
 		k3 = 1
 		A = np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
 		G = A
 		
 		L = csgraph.laplacian(G, normed = False)
 		I = np.identity(n)
 		W = I - epsilon * L  # W is a double stochastic matrix
 		#W = np.linalg.inv(W.T)
 		#W = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=featureDimension))) )
 		print W
		return W.T

	def initializeGW(self, Gepsilon):
		n = len(self.users)

		a = np.ones(n-1) 
 		b =np.ones(n);
 		c = np.ones(n-1)
 		k1 = -1
 		k2 = 0
 		k3 = 1
 		A = np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
 		G = A
 		
 		L = csgraph.laplacian(G, normed = False)
 		I = np.identity(n)
 		GW = I + Gepsilon*L  # W is a double stochastic matrix
 		print GW
		return GW.T

	def getW(self):
		return self.W
	def getGW(self):
		return self.GW

	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self):
		self.articlePool = sample(self.articles, self.poolArticleSize)

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.dimension)
			for uj in self.users:
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)
			print 'Users', ui.id, 'CoTheta', ui.CoTheta

	def getReward(self, user, pickedArticle):
		return np.dot(user.CoTheta, pickedArticle.featureVector)

	def GetOptimalReward(self, user, articlePool):		
		maxReward = sys.float_info.min
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
		return maxReward
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def runAlgorithms(self, algorithms):
		# get cotheta for each user
		self.CoTheta()
		self.startTime = datetime.datetime.now()

		tim_ = []
		BatchAverageRegret = {}
		AccRegret = {}
		ThetaDiffList = {}
		CoThetaDiffList = {}
		
		ThetaDiffList_user = {}
		CoThetaDiffList_user = {}
		
		# Initialization
		for alg_name in algorithms.iterkeys():
			BatchAverageRegret[alg_name] = []
			ThetaDiffList[alg_name] = []
			CoThetaDiffList[alg_name] = []
			AccRegret[alg_name] = {}

			for i in range(len(self.users)):
				AccRegret[alg_name][i] = []
		
		userSize = len(self.users)
		# Loop begin
		for iter_ in range(self.iterations):
			# prepare to record theta estimation error
			for alg_name in algorithms.iterkeys():
				ThetaDiffList_user[alg_name] = []
				CoThetaDiffList_user[alg_name] = []
				
			for u in self.users:
				self.regulateArticlePool() # select random articles

				noise = self.noise()
				#get optimal reward for user x at time t
				OptimalReward = self.GetOptimalReward(u, self.articlePool) + noise
							
				for alg_name, alg in algorithms.items():
					pickedArticle = alg.decide(self.articlePool, u.id)
					reward = self.getReward(u, pickedArticle) + noise
					alg.updateParameters(pickedArticle, reward, u.id)

					regret = OptimalReward - reward	
					AccRegret[alg_name][u.id].append(regret)

					# every algorithm will estimate co-theta
					
					if alg_name == 'CoLinUCB' or alg_name == 'syncCoLinUCB' or alg_name == 'AsyncCoLinUCB':
						CoThetaDiffList_user[alg_name] += [self.getL2Diff(u.CoTheta, alg.getCoThetaFromCoLinUCB(u.id))]
						ThetaDiffList_user[alg_name] += [self.getL2Diff(u.theta, alg.getLearntParameters(u.id))]		
					elif alg_name == 'LinUCB'  or alg_name == 'GOBLin':
						CoThetaDiffList_user[alg_name] += [self.getL2Diff(u.CoTheta, alg.getLearntParameters(u.id))]
			for alg_name, alg in algorithms.items():
				if alg_name == 'syncCoLinUCB':
					alg.LateUpdate()						

			for alg_name in algorithms.iterkeys():
				CoThetaDiffList[alg_name] += [sum(CoThetaDiffList_user[alg_name])/userSize]
				if alg_name == 'CoLinUCB' or alg_name == 'syncCoLinUCB':
					ThetaDiffList[alg_name] += [sum(ThetaDiffList_user[alg_name])/userSize]
				
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					TotalAccRegret = sum(sum (u) for u in AccRegret[alg_name].itervalues())
					BatchAverageRegret[alg_name].append(TotalAccRegret)
		
		# plot the results		
		f, axa = plt.subplots(2, sharex=True)
		# plot regard
		for alg_name in algorithms.iterkeys():		
			axa[0].plot(tim_, BatchAverageRegret[alg_name], label = alg_name)
			axa[0].lines[-1].set_linewidth(1.5)
			print '%s: %.2f' % (alg_name, BatchAverageRegret[alg_name][-1])
		axa[0].legend()
		axa[0].set_xlabel("Iteration")
		axa[0].set_ylabel("Regret")
		axa[0].set_title("Noise scale = " + str(self.NoiseScale))
		
		# plot the estimation error of co-theta
		time = range(self.iterations)
		for alg_name in algorithms.iterkeys():
			axa[1].plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
			axa[1].lines[-1].set_linewidth(1.5)	
		axa[1].legend()
		axa[1].set_xlabel("Iteration")
		axa[1].set_ylabel("L2 Diff")
		axa[1].set_yscale('log')
		'''
		
		# plot the estimation error of theta
		for alg_name in algorithms.iterkeys():
			if alg_name == 'CoLinUCB' or alg_name == 'syncCoLinUCB':
				axa[2].plot(time, CoThetaDiffList[alg_name], label = alg_name + '_Theta')
				axa[2].lines[-1].set_linewidth(1.5)			
		axa[2].legend()
		axa[2].set_xlabel("Iteration")
		axa[2].set_ylabel("L2 Diff")
		axa[2].set_yscale('log')
		'''
		
		plt.show()


if __name__ == '__main__':
	iterations = 500
	NoiseScale = .01

	dimension = 5
	alpha  = 0.2 
	lambda_ = 0.2   # Initialize A
	epsilon = 0.4 # initialize W

	n_articles = 1000
	ArticleGroups = 5

	n_users = 10
	UserGroups = 5	

	poolSize = 10
	batchSize = 10

	# Parameters for GOBLin
	G_alpha = .2
	G_lambda_ = 0.2
	Gepsilon = 0.4
	# Epsilon_greedy parameter
	eGreedy = 0.3
	
	userFilename = os.path.join(sim_files_folder, "users_"+str(n_users)+"+dim-"+str(dimension)+ "Ugroups" + str(UserGroups)+".json")
	
	#"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	# we can choose to simulate users every time we run the program or simulate users once, save it to 'sim_files_folder', and keep using it.
	UM = UserManager(dimension, n_users, UserGroups = UserGroups, thetaFunc=featureUniform, argv={'l2_limit':1})
	#users = UM.simulateThetafromUsers()
	#UM.saveUsers(users, userFilename, force = False)
	users = UM.loadUsers(userFilename)

	articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles)+"+dim"+str(dimension) + "Agroups" + str(ArticleGroups)+".json")
	# Similarly, we can choose to simulate articles every time we run the program or simulate articles once, save it to 'sim_files_folder', and keep using it.
	AM = ArticleManager(dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
			FeatureFunc=featureUniform,  argv={'l2_limit':1})
	#articles = AM.simulateArticlePool()
	#AM.saveArticles(articles, articlesFilename, force=False)
	articles = AM.loadArticles(articlesFilename)

	simExperiment = simulateOnlineData(dimension  = dimension,
						iterations = iterations,
						articles=articles,
						users = users,		
						noise = lambda : np.random.normal(scale = NoiseScale),
						batchSize = batchSize,
						type_ = "UniformTheta", 
						signature = AM.signature,
						poolArticleSize = poolSize, NoiseScale = NoiseScale, epsilon = epsilon, Gepsilon =Gepsilon)
	print "Starting for ", simExperiment.simulation_signature

	algorithms = {}
	algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	algorithms['GOBLin'] = GOBLinAlgorithm( dimension= dimension, alpha = G_alpha, lambda_ = G_lambda_, n = n_users, W = simExperiment.getGW() )
	algorithms['syncCoLinUCB'] = syncCoLinUCBAlgorithm(dimension=dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
	algorithms['AsyncCoLinUCB'] = AsyCoLinUCBAlgorithm(dimension=dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())

	algorithms['eGreedy'] = eGreedyAlgorithm(epsilon = eGreedy)
	algorithms['UCB1'] = UCB1Algorithm()
	
	
	simExperiment.runAlgorithms(algorithms)



	
