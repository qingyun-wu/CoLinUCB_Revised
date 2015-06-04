from conf import * 	# it saves the address of data stored and where to save the data produced by algorithms
import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter  	# for easiness in sorting and finding max and stuff
import datetime
import numpy as np 	# many operations are done in numpy as matrix inverse; for efficiency
from scipy.sparse import csgraph 
from scipy.spatial import distance
from scipy.linalg import sqrtm


# time conventions in file name
# dataDay + Month + Day + Hour + Minute

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
 
def getArticles(fileNameRead):
    with open(fileNameRead, 'r') as f:
        ArticleIDs = []
        ArticleFeatures = []
        for line in f:
            vec = []
            line = line.split(';')
            
            ArticleIDs.append(float(line[0]))
            #print line
    
            word = line[1].split('  ')
            if len(word)==5:
	            for i in range(5):
	                vec.append(float(word[i]))
	            ArticleFeatures.append(np.asarray(vec))
    	ArticleFeatures = np.asarray(ArticleFeatures)
    #print  ArticleFeatures
    return ArticleIDs, ArticleFeatures

def initializeGW(ArticleFeatureVectors, Gepsilon):
	n = len(ArticleFeatureVectors)
	W = np.zeros(shape = (n, n))
		
	for i in range(n):
		sSim = 0
		for j in range(n):
			sim = np.dot(ArticleFeatureVectors[i],ArticleFeatureVectors[j])
			print 'sim',sim
			if i == j:
				sim += 1
			W[i][j] = sim
			sSim += sim
			
		W[i] /= sSim
		#for a in range(n):
		#	print '%.3f' % W[i][a],
		#print ''
	G = W
	L = csgraph.laplacian(G, normed = False)
	I = np.identity(n)
	GW = I + Gepsilon*L  # W is a double stochastic matrix
	print GW          
	return GW.T


# data structures for different strategies and parameters
# data structure to store ctr	
class articleAccess():
	def __init__(self):
		self.accesses = 0.0 # times the article was chosen to be presented as the best articles
		self.clicks = 0.0 	# of times the article was actually clicked by the user
		self.CTR = 0.0 		# ctr as calculated by the updateCTR function

	def updateCTR(self):
		try:
			self.CTR = self.clicks / self.accesses
		except ZeroDivisionError: # if it has not been accessed
			self.CTR = -1
		return self.CTR

	def addrecord(self, click):
		self.clicks += click
		self.accesses += 1

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
	def __init__(self):
		self.learn_stats = articleAccess()
		self.deploy_stats = articleAccess()

class LinUCBStruct:
	def __init__(self, lambda_,  d, articleID):
		self.d = d
		self.articleID = articleID
		self.A = lambda_*np.identity(n = d)
		self.b = np.zeros(d)
		self.theta = np.zeros(d)

		self.learn_stats = articleAccess()

	def updateParameters(self, userFeatureVector, reward):
		self.A +=np.outer(userFeatureVector, userFeatureVector)
		self.b += userFeatureVector*reward

		self.theta = np.dot(np.linalg.inv(self.A), self.b)
	def getLinUCBPta(self, alpha, userFeatureVector):
		mean = np.dot(self.theta, userFeatureVector)
		var = np.sqrt(np.dot( np.dot(userFeatureVector, np.linalg.inv(self.A)) , userFeatureVector))

		pta = mean + alpha*var
		return pta

class GOBLinStruct:
	def __init__(self, lambda_, d,  ArticleFeatures, Gepsilon):
		self.d = d
		self.articleNum= len(ArticleFeatures)
		self.learn_stats = articleAccess()	# in paper the evaluation is done on two buckets; so the stats are saved for both of them separately; In this code I am not doing deployment, so the code learns on all examples

		self.W = initializeGW(ArticleFeatures, Gepsilon)
		self.A = lambda_* np.identity(n = d*self.articleNum)
		self.b = np.zeros(d*self.articleNum)
		
		self.theta = np.dot(np.linalg.inv(self.A), self.b)
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(self.W, np.identity(n=d))) )
		self.STBigW = sqrtm(np.kron(self.W, np.identity(n=d)))
	def updateParameters(self, userFeatureVector, reward, articleID):
		featureVectorV = np.zeros(self.d*self.articleNum)
		featureVectorV[float(articleID)*d:(float(articleID)+1)*d] = np.asarray(userFeatureVector)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		self.A += np.outer(CoFeaV, CoFeaV)
		self.b += click * CoFeaV

		self.theta = np.dot(np.linalg.inv(self.A), self.b)

	def getGOBLinPta(self, alpha, userfeatureVector, articleID):

		featureVectorV = np.zeros(self.d*self.articleNum)
		featureVectorV[float(articleID)*d:(float(articleID)+1)*d] = np.asarray(userfeatureVector)
		
		CoFeaV = np.dot(self.STBigWInv, featureVectorV)

		mean = np.dot(np.transpose(self.theta), CoFeaV)
	
		var = np.sqrt( np.dot( np.dot(CoFeaV, np.linalg.inv(self.A)) , CoFeaV))
		pta = mean + alpha * var
		return pta
	

# This code simply reads one line from the source files of Yahoo!. Please see the yahoo info file to understand the format. I tested this part; so should be good but second pair of eyes could help
def parseLine(line):
	line = line.split("|")

	tim, articleID, click = line[0].strip().split(" ")
	tim, articleID, click = int(tim), int(articleID), int(click)
	# user_features = np.array([x for ind,x in enumerate(re.split(r"[: ]", line[1])) if ind%2==0][1:])
	user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])

	pool_articles = [l.strip().split(" ") for l in line[2:]]
	pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
	return tim, articleID, click, user_features, pool_articles

def save_to_file(fileNameWrite, recordedStats, tim):
	with open(fileNameWrite, 'a+') as f:
		f.write('data') # the observation line starts with data;
		f.write(',' + str(tim))
		f.write(',' + ';'.join([str(x) for x in recordedStats]))
		f.write('\n')
    
# the first thing that executes in this programme is this main function
if __name__ == '__main__':
	# I regularly print stuff to see if everything is going alright. Its usually helpfull if i change code and see if i did not make obvious mistakes
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():

		
	
		randomCTR = articles_random.learn_stats.updateCTR()
		
		GOBLinCTR = GOBLin_ART.learn_stats.updateCTR()

		TotalLinUCBAccess = 0.0
		TotalLinUCBClick = 0.0
		for i in range(articleNum):			
			TotalLinUCBAccess += LinUCB_arts[i].learn_stats.accesses
			TotalLinUCBClick += LinUCB_arts[i].learn_stats.clicks
		#print 'TotalLinUCBAccess ', TotalLinUCBAccess, '  LinUCBAccess ', LinUCBAccess 
		#print 'TotalLinUCBClick ', TotalLinUCBClick, '  LinUCBClick ', LinUCBClick
		if TotalLinUCBAccess != 0:
			LinUCBCTR = TotalLinUCBClick/(1.0*TotalLinUCBAccess)
		else:
			LinUCBCTR = -1.0


		print totalObservations
		#if randomLearnCTR != 0:
		print 'random', randomCTR,'  GoBLin', GOBLinCTR,  '	LinUCB', LinUCBCTR	

		recordedStats = [randomCTR, GOBLinCTR, LinUCBCTR]
		save_to_file(fileNameWrite, recordedStats, tim) 


	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	fileSig = 'GoBLin'
	dataDay = '01'
	#fileName = yahoo_address + '/Test'
	fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	
	fileNameWrite = os.path.join(save_address, fileSig + timeRun + '.csv')

	fileNameArticles =   os.path.join(data_address, 'AllArticles1.txt') 
	ArticleIDs,ArticleFeatureVectors = getArticles(fileNameArticles)   # get all articleIDs and all FeatureVectors
	articleNum = len(ArticleIDs)
	#print ArticleFeatureVectors, len(ArticleFeatureVectors)
							
	batchSize = 200							# size of one batch
	testEnviornment = False

	d = 5 	
	alpha = 0.3										# dimension of the input sizes
	lambda_ = 0.2
	articles_random = randomStruct()
	
	userNum = 10	
	countLine =-1
	totalObservations = 0
	totalReward = 0
	epsilon = 1.0

	GOBLin_ART = GOBLinStruct(lambda_, d, ArticleFeatureVectors, epsilon)
	LinUCB_arts = []
	for i in range(articleNum):
		LinUCB_arts.append(LinUCBStruct( lambda_, d, i ))

	# put some new data in file for readability
	with open(fileNameWrite, 'a+') as f:
		f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
		# format style, '()' means that it is repeated for each article
		f.write('\n, Time,RandomCTR;CoLinUCBCTR;LinUCBCTR; CoLinUCBParameterDif; LinUCBParametersDif\n')

	print fileName, fileNameWrite
	

	with open(fileName, 'r') as f:
		# reading file line ie observations running one at a time
		for line in f:
			countLine = countLine + 1
			totalObservations +=1
			#currentUserID = userIDAssignment[countLine]

			tim, article_chosen, click, user_features, pool_articles = parseLine(line)
			featureVector = user_features[:-1]

			#currentUserID = getIDAssignment(np.asarray(featureVector), userFeatureVectors)
			currentArticles = []

			GOBLin_maxPTA = float('-inf')
			LinUCB_maxPTA = float('-inf')

			GOBLinPicked = None
			LinUCBPicked = None

			for article in pool_articles:	
				article_id = float(article[0])
				article_index = ArticleIDs.index(article_id)
				article_featureVector = article[1:6]
				currentArticles.append(article_id)
				if len(article_featureVector) == 5:
					# CoLinUCB pick article
					GOBLin_pta = GOBLin_ART.getGOBLinPta(alpha, featureVector, article_index )
					# pick article with highest Prob
					if GOBLin_maxPTA <GOBLin_pta:
						GOBLinPicked = article_id    # article picked by CoLinUCB
						GOBLin_PickedfeatureVector = article_featureVector
						GOBLinPickedIndex = ArticleIDs.index(GOBLinPicked)
						GOBLin_maxPTA = GOBLin_pta

					# LinUCB pick article
					LinUCB_pta = LinUCB_arts[article_index ].getLinUCBPta(alpha, featureVector)
					if LinUCB_maxPTA < LinUCB_pta:
						LinUCBPicked = article_id    # article picked by CoLinUCB
						LinUCB_PickedfeatureVector = article_featureVector
						LinPickedIndex = ArticleIDs.index(LinUCBPicked)
						LinUCB_maxPTA = LinUCB_pta

				
			# article picked by random strategy
			randomArticle = choice(currentArticles)

			# if random strategy article Picked by evaluation srategy
			if randomArticle == article_chosen:
				articles_random.learn_stats.addrecord(click)

			if GOBLinPicked == article_chosen:
				GOBLin_ART.learn_stats.addrecord(click)
				GOBLin_ART.updateParameters(featureVector, click, GOBLinPickedIndex)
				
			if LinUCBPicked == article_chosen:
				LinUCB_arts[LinPickedIndex].learn_stats.addrecord(click)
				LinUCB_arts[LinPickedIndex].updateParameters(featureVector, click)
				
				
			# if the batch has ended 
			if totalObservations%batchSize==0:
				# write observations for this batch
				printWrite()
					

		# print stuff to screen and save parameters to file when the Yahoo! dataset file ends
		printWrite()
		
