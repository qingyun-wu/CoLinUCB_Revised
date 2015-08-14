import numpy as np
from random import sample
from scipy.sparse import csgraph
from datetime import datetime 
import os.path
import matplotlib.pyplot as plt
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform
from Articles import ArticleManager
from Users import UserManager
from LinUCB import LinUCBAlgorithm
from GOBLin import GOBLinAlgorithm
from CoLin import AsyCoLinUCBAlgorithm, syncCoLinUCBAlgorithm

class simulateOnlineData(object):
	def __init__(self, dimension, iterations, articles, users, 
					batchSize = 1000,
					noise = lambda : 0,
					matrixNoise = lambda:0,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					NoiseScale = 0,
					sparseLevel = 0, # 0 means dense graph
					epsilon = 1, Gepsilon = 1):

		self.simulation_signature = signature
		self.type = type_

		self.dimension = dimension
		self.iterations = iterations
		self.noise = noise
		self.matrixNoise = matrixNoise # noise to be added to W
		self.articles = articles 
		self.users = users
		self.sparseLevel = sparseLevel

		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		self.W = self.initializeW(epsilon)
		self.GW = self.initializeGW(Gepsilon)
		self.NoiseScale = NoiseScale
		
	def constructAdjMatrix(self):
		n = len(self.users)	

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			sSim = 0
			for uj in self.users:
				sim = np.dot(ui.theta, uj.theta) + self.matrixNoise() # is dot product sufficient
				if sim < 0:
					sim = 0
				G[ui.id][uj.id] = sim
				sSim += sim
				
			G[ui.id] /= sSim
		return G

	def constructSparseMatrix(self, m):
		n = len(self.users)	
		m = min(n, m); # in case m is set too large

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			# construct the original connections
			for uj in self.users:
				sim = np.dot(ui.theta, uj.theta) + self.matrixNoise() # is dot product sufficient
				if sim < 0:
					sim = 0
				G[ui.id][uj.id] = sim
			
			# find out the top M similar users
			similarity = sorted(G[ui.id], reverse=True)
			threshold = similarity[m]
			simSum = sum(similarity[0:m])
			
			# trim and renormalize the graph
			for i in range(n):
				if G[ui.id][i] > threshold:
					G[ui.id][i] /= simSum
				else:
					G[ui.id][i] = 0;
		return G

	# create user connectivity graph
	def initializeW(self, epsilon):	
		if self.sparseLevel > 0:
			W = self.constructSparseMatrix(self.sparseLevel)   # sparse matrix top m users 
		else:
			W = self.constructAdjMatrix()
		print 'W.T', W.T
		return W.T

	def initializeGW(self, Gepsilon):
		if self.sparseLevel > 0:
			G = self.constructSparseMatrix(self.sparseLevel)
		else:
			G = self.constructAdjMatrix()	
			
		L = csgraph.laplacian(G, normed = False)
		I = np.identity(n = G.shape[0])
		GW = I + Gepsilon*L  # W is a double stochastic matrix
		print 'GW', GW
		return GW.T

	def getW(self):
		return self.W
	
	def getGW(self):
		return self.GW

	def getTheta(self):
		Theta = np.zeros(shape = (self.dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.dimension)
			for uj in self.users:
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)
			print 'Users', ui.id, 'CoTheta', ui.CoTheta	
	
	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self):
		# Randomly generate articles
		self.articlePool = sample(self.articles, self.poolArticleSize)   

	def getReward(self, user, pickedArticle):
		return np.dot(user.CoTheta, pickedArticle.featureVector)

	def GetOptimalReward(self, user, articlePool):		
		maxReward = float('-inf')
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
		return maxReward
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')

		# compute co-theta for every user
		self.CoTheta()

		tim_ = []
		BatchCumlateRegret = {}
		AlgRegret = {}
		ThetaDiffList = {}
		CoThetaDiffList = {}
		WDiffList = {}
		
		ThetaDiff = {}
		CoThetaDiff = {}
		WDiff = {}
		
		# Initialization
		userSize = len(self.users)
		for alg_name in algorithms.iterkeys():
			AlgRegret[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			
			if alg_name in ['syncCoLinUCB', 'AsyncCoLinUCB', 'WCoLinUCB', 'WknowTheta', 'W_W0']:
				ThetaDiffList[alg_name] = []
				CoThetaDiffList[alg_name] = []		
			if alg_name in ['WCoLinUCB', 'WknowTheta', 'W_W0']:
				WDiffList[alg_name] = []
		
		with open(filenameWriteRegret, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.iterkeys()]))
			f.write('\n')
		with open(filenameWritePara, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name)+'CoTheta' for alg_name in algorithms.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'W' for alg_name in WDiffList.iterkeys()]))
			f.write('\n')
		
		# Loop begin
		for iter_ in range(self.iterations):
			# prepare to record theta estimation error
			for alg_name in algorithms.iterkeys():
				if alg_name in ['syncCoLinUCB', 'AsyncCoLinUCB', 'WCoLinUCB', 'WknowTheta', 'W_W0']:
					ThetaDiff[alg_name] = 0
					CoThetaDiff[alg_name] = 0
				if alg_name in ['WCoLinUCB', 'WknowTheta', 'W_W0']:
					WDiff[alg_name] = 0
					
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
					AlgRegret[alg_name].append(regret)

					# every algorithm will estimate co-theta					
					if  alg_name in ['syncCoLinUCB', 'AsyncCoLinUCB', 'WCoLinUCB', 'WknowTheta', 'W_W0']:
						CoThetaDiff[alg_name] += self.getL2Diff(u.CoTheta, alg.getCoThetaFromCoLinUCB(u.id))
						ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getLearntParameters(u.id))
						if alg_name in ['WCoLinUCB', 'WknowTheta', 'W_W0']:
							WDiff[alg_name] += self.getL2Diff(self.W.T[u.id], alg.getW(u.id))	
					elif alg_name in ['LinUCB', 'GOBLin']:
						CoThetaDiff[alg_name] += self.getL2Diff(u.CoTheta, alg.getLearntParameters(u.id))
			
			if 'syncCoLinUCB' in algorithms:
				algorithms['syncCoLinUCB'].LateUpdate()						

			for alg_name in algorithms.iterkeys():
				CoThetaDiffList[alg_name] += [CoThetaDiff[alg_name]/userSize]
				if alg_name in ['syncCoLinUCB', 'AsyncCoLinUCB']:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name]/userSize]
				if alg_name in ['WCoLinUCB', 'WknowTheta', 'W_W0']:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name]/userSize]
					WDiffList[alg_name] += [WDiff[alg_name]/userSize]
				
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))
				
				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
					f.write('\n')
				with open(filenameWritePara, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(CoThetaDiffList[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
					f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.iterkeys()]))
					f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in WDiffList.iterkeys()]))
					f.write('\n')
					
		# plot the results		
		f, axa = plt.subplots(2, sharex=True)
		for alg_name in algorithms.iterkeys():	
			axa[0].plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
			print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])
		axa[0].legend(loc='lower right',prop={'size':9})
		axa[0].set_xlabel("Iteration")
		axa[0].set_ylabel("Regret")
		axa[0].set_title("Accumulated Regret")
		
		# plot the estimation error of co-theta
		time = range(self.iterations)
		
		for alg_name in algorithms.iterkeys():
			axa[1].plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
			if alg_name == 'AsyncCoLinUCB' or alg_name =='syncCoLinUCB' or alg_name =='WCoLinUCB' or alg_name =='W_W0':
				axa[1].plot(time, ThetaDiffList[alg_name], label = alg_name + '_Theta')
		
		axa[1].legend(loc='upper right',prop={'size':6})
		axa[1].set_xlabel("Iteration")
		axa[1].set_ylabel("L2 Diff")
		axa[1].set_yscale('log')
		axa[1].set_title("Parameter estimation error")
		plt.show()

if __name__ == '__main__':
	iterations = 500
	NoiseScale = .1
	matrixNoise = .3

	dimension = 5
	alpha  = 0.2
	lambda_ = 0.1   # Initialize A
	epsilon = 0 # initialize W
	eta_ = 0.1

	n_articles = 1000
	ArticleGroups = 5

	n_users = 10
	UserGroups = 5	

	poolSize = 10
	batchSize = 10

	# Parameters for GOBLin
	G_alpha = alpha
	G_lambda_ = lambda_
	Gepsilon = 1
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
						matrixNoise = lambda : np.random.normal(scale = matrixNoise),
						batchSize = batchSize,
						type_ = "UniformTheta", 
						signature = AM.signature,
						poolArticleSize = poolSize, NoiseScale = NoiseScale, epsilon = epsilon, Gepsilon =Gepsilon)

	print "Starting for ", simExperiment.simulation_signature

	algorithms = {}
	
	algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	algorithms['GOBLin'] = GOBLinAlgorithm( dimension= dimension, alpha = G_alpha, lambda_ = G_lambda_, n = n_users, W = simExperiment.getGW0() )
	algorithms['syncCoLinUCB'] = syncCoLinUCBAlgorithm(dimension=dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW0())
	algorithms['AsyncCoLinUCB'] = AsyCoLinUCBAlgorithm(dimension=dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW0())
	
	#algorithms['WCoLinUCB'] =  WAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, eta_ = eta_, n = n_users)
	#algorithms['WknowTheta'] = WknowThetaAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, eta_ = eta_, n = n_users, theta = simExperiment.getTheta())
	#algorithms['W_W0'] = W_W0_Algorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, eta_ = eta_, n = n_users, W0 = simExperiment.getW0())

	#algorithms['eGreedy'] = eGreedyAlgorithm(epsilon = eGreedy)
	#algorithms['UCB1'] = UCB1Algorithm()
	
	simExperiment.runAlgorithms(algorithms)