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
from util_functions import getClusters, getIDAssignment, parseLine, save_to_file, initializeGW


# time conventions in file name
# dataDay + Month + Day + Hour + Minute


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

class GOBLinStruct:
	def __init__(self, lambda_, d,  Gepsilon, userNum, userFeatureVectors):
		self.learn_stats = articleAccess()	

		self.d = d
		self.userNum= userNum

		self.W = initializeGW(userFeatureVectors, Gepsilon)

		self.A = lambda_* np.identity(n = d*self.userNum)
		self.b = np.zeros(d*self.userNum)
                self.AInv = np.linalg.inv(self.A)
		
		self.theta = np.dot(np.linalg.inv(self.A), self.b)   # Long vector
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(self.W, np.identity(n=d))) )
		self.STBigW = sqrtm(np.kron(self.W, np.identity(n=d)))
	def updateParameters(self, PickedfeatureVector, reward,userID):
		featureVectorV = np.zeros(self.d*self.userNum)
		featureVectorV[float(userID)*d:(float(userID)+1)*d] = np.asarray(PickedfeatureVector)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		self.A += np.outer(CoFeaV, CoFeaV)
		self.b += click * CoFeaV

                self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv, self.b)

	def getGOBLinPta(self, alpha, featureVector, userID):

		featureVectorV = np.zeros(self.d*self.userNum)
		featureVectorV[float(userID)*d:(float(userID)+1)*d] = np.asarray(featureVector)
		
		CoFeaV = np.dot(self.STBigWInv, featureVectorV)

		mean = np.dot(np.transpose(self.theta), CoFeaV)
	
		var = np.sqrt( np.dot( np.dot(CoFeaV, self.AInv) , CoFeaV))
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
		GOBLinCTR = GOBLin_USERS.learn_stats.updateCTR()

		print totalObservations
		#if randomLearnCTR != 0:
		print 'random', randomCTR,'  GoBLin', GOBLinCTR  	

		recordedStats = [randomCTR, GOBLinCTR]
		save_to_file(fileNameWrite, recordedStats, tim) 

	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
	fileSig = 'GOB_20_'
	batchSize = 2000							# size of one batch
	
	d = 5 	        # feature dimension
	alpha = 0.3     # control how much to explore
	lambda_ = 0.2   # regularization used in matrix A
	epsilon = 0.3
    
	totalObservations = 0

	fileNameWriteCluster = os.path.join(data_address, 'kmeans_model_20.dat')
	userFeatureVectors = getClusters(fileNameWriteCluster)	
	userNum = len(userFeatureVectors)
	
	articles_random = randomStruct()
	GOBLin_USERS = GOBLinStruct(lambda_, d, epsilon, userNum,userFeatureVectors)

	for dataDay in dataDays:
		fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	
		fileNameWrite = os.path.join(save_address, fileSig + dataDay + timeRun + '.csv')

		# put some new data in file for readability
		with open(fileNameWrite, 'a+') as f:
			f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
			f.write('\n, Time,RandomCTR;GOBLinCTR\n')

		print fileName, fileNameWrite
	

		with open(fileName, 'r') as f:
			# reading file line ie observations running one at a time
			for line in f:
				totalObservations +=1

				tim, article_chosen, click, user_features, pool_articles = parseLine(line)
				currentUser_featureVector = user_features[:-1]

				currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)                
                
                                #-----------------------------Pick an article (CoLinUCB, LinUCB, Random)-------------------------
                                currentArticles = []
                                GOBLin_maxPTA = float('-inf')
                                GOBLinPicked = None
                                GOBLin_PickedfeatureVector = np.array([0,0,0,0,0])
                                for article in pool_articles:
                                        article_id = article[0]
                                        article_featureVector =np.asarray(article[1:6])
                                        currentArticles.append(article_id)
                                        if len(article_featureVector)==5:
                                                GOBLin_pta = GOBLin_USERS.getGOBLinPta(alpha, article_featureVector, currentUserID)
                                                if GOBLin_maxPTA < GOBLin_pta:
                                                        GOBLinPicked = article_id    # article picked by GOB.Lin
                                                        GOBLin_PickedfeatureVector = article_featureVector
                                                        GOBLin_maxPTA = GOBLin_pta

                                #Record log CTR
                                articles_random.learn_stats.addrecord(click)
                                #Update parameter if chosen article matchs the logged article
                                if GOBLinPicked == article_chosen:
                                        GOBLin_USERS.learn_stats.addrecord(click)
                                        GOBLin_USERS.updateParameters(GOBLin_PickedfeatureVector, click, currentUserID)

                                # if the batch has ended
                                if totalObservations%batchSize==0:
                                        # write observations for this batch
                                        printWrite()
                        #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
                        printWrite()

