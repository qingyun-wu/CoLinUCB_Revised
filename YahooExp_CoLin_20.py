from conf import * 	# it saves the address of data stored and where to save the data produced by algorithms
import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import datetime
import numpy as np 	
from scipy.sparse import csgraph
from scipy.spatial import distance
from YahooExp_util_functions import getClusters, getIDAssignment, parseLine, save_to_file, initializeW, vectorize, matrixize, articleAccess


from CoLin import AsyCoLinUCBUserSharedStruct, AsyCoLinUCBAlgorithm, CoLinUCBUserSharedStruct

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
	def __init__(self):
		self.learn_stats = articleAccess()

# structure to save data from CoLinUCB strategy
class CoLinUCBStruct(AsyCoLinUCBUserSharedStruct):
	def __init__(self, featureDimension, lambda_, userNum, W):
		AsyCoLinUCBUserSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = userNum, W = W)
		self.learn_stats = articleAccess()	

if __name__ == '__main__':
	# regularly print stuff to see if everything is going alright.
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():
		randomLearnCTR = articles_random.learn_stats.updateCTR()	
		CoLinUCBCTR = CoLinUCB_USERS.learn_stats.updateCTR()

		print totalObservations
		print 'random', randomLearnCTR,'  CoLin', CoLinUCBCTR
		
		recordedStats = [randomLearnCTR, CoLinUCBCTR]
		# write to file
		save_to_file(fileNameWrite, recordedStats, tim) 


	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
	fileSig = 'CoLin_20_'
	batchSize = 2000							# size of one batch
	
	d = 5 	        # feature dimension
	alpha = 0.3     # control how much to explore
	lambda_ = 0.2   # regularization used in matrix A
    
	totalObservations = 0

	fileNameWriteCluster = os.path.join(data_address, 'kmeans_model_20.dat')
	userFeatureVectors = getClusters(fileNameWriteCluster)	
	userNum = len(userFeatureVectors)
	W = initializeW(userFeatureVectors)   # Generate user relation matrix
	
	articles_random = randomStruct()
	CoLinUCB_USERS = CoLinUCBStruct(d, lambda_ ,userNum, W)
	

	for dataDay in dataDays:
		fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	
		fileNameWrite = os.path.join(Yahoo_save_address, fileSig + dataDay + timeRun + '.csv')

		# put some new data in file for readability
		with open(fileNameWrite, 'a+') as f:
			f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
			f.write('\n, Time,RandomCTR;CoLinUCBCTR\n')

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
                                CoLinUCB_maxPTA = float('-inf')
                                CoLinUCBPicked = None      
                                #CoLinUCB_PickedfeatureVector = np.array([0,0,0,0,0])
                               
                                for article in pool_articles:
                                        article_id = article[0]
                                        article_featureVector =np.asarray(article[1:6])
                                        currentArticles.append(article_id)
                                        # CoLinUCB pick article
                                        if len(article_featureVector)==5:
                                                CoLinUCB_pta = CoLinUCB_USERS.getProb(alpha, article_featureVector, currentUserID)
                                                if CoLinUCB_maxPTA < CoLinUCB_pta:
                                                        CoLinUCBPicked = article_id    # article picked by CoLinUCB
                                                        CoLinUCB_PickedfeatureVector = article_featureVector
                                                        CoLinUCB_maxPTA = CoLinUCB_pta
                               

                                # article picked by random strategy
                                articles_random.learn_stats.addrecord(click)
                                if CoLinUCBPicked == article_chosen:
                                        CoLinUCB_USERS.learn_stats.addrecord(click)
                                        CoLinUCB_USERS.updateParameters(CoLinUCB_PickedfeatureVector, click, currentUserID)

                                # if the batch has ended
                                if totalObservations%batchSize==0:
                                        printWrite()
                        #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
                        printWrite()
