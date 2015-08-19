'''
Created on Aug 13, 2015

@author: hongning
'''
from Users import UserManager
from Articles import ArticleManager
from Simulation import simulateOnlineData
from util_functions import featureUniform
import datetime
from conf import save_address, sim_files_folder
from os import path
from random import choice
import matplotlib.pyplot as plt
import numpy as np
from GOBLin import GOBLin_SelectUserAlgorithm
from CoLin import AsyCoLinUCBAlgorithm, CoLinUCB_SelectUserAlgorithm
from LinUCB import LinUCB_SelectUserAlgorithm

class simulateOnlineData_SelectUser(simulateOnlineData):
    def regulateArticlePool(self, iter_):        
        #generate article pool regularly in order to get rid of randomness
        if (iter_+1)*self.poolArticleSize > len(self.articles):
            a = (iter_+1*self.poolArticleSize)%len(self.articles)/self.poolArticleSize 
            b = (iter_+1*self.poolArticleSize)%len(self.articles)%self.poolArticleSize 
            iter_ = 10*(a%10)+b 
        self.articlePool = self.articles[iter_* self.poolArticleSize : (iter_+1)*self.poolArticleSize]

    def GetOptimalUserReward(self, AllUsers, articlePool):
        maxReward = float('-inf')
        OptimalUser = None
        OptimalArticle = None
        for x in articlePool:
            for u in AllUsers:
                reward = self.getReward(u,x)
                if reward > maxReward:
                    maxReward = reward
                    OptimalUser = u
                    OptimalArticle = x
        return OptimalUser, OptimalArticle, maxReward

    def runAlgorithms(self, algorithms):
        # get cotheta for each user
        self.startTime = datetime.datetime.now()
        timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
        filenameWriteRegret = path.join(save_address, 'AccRegret' + timeRun + '.csv')
        filenameWritePara = path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')

        self.CoTheta()

        tim_ = []
        BatchAverageRegret = {}
        AccRegret = {}
        ThetaDiffList = {}
        CoThetaDiffList = {}
        WDiffList = {}
        
        ThetaDiffList_user = {}
        CoThetaDiffList_user = {}
        WDiffList_user = {}

        ThetaDiff = {}
        CoThetaDiff = {}
        WDiff = {}
        
        # Initialization
        userSize = len(self.users)
        '''
        for alg_name, alg in algorithms.items():
            BatchAverageRegret[alg_name] = []
            AccRegret[alg_name] = {}
            if alg.CanEstimateUserPreference:
                ThetaDiffList[alg_name] = []
            if alg.CanEstimateCoUserPreference:
                CoThetaDiffList[alg_name] = []
            if alg.CanEstimateW:
                WDiffList[alg_name] = []

            for i in range(userSize):
                AccRegret[alg_name][i] = []
        '''
        for alg_name in algorithms.iterkeys():
            BatchAverageRegret[alg_name] = []
            
            CoThetaDiffList[alg_name] = []
            AccRegret[alg_name] = {}
            if alg_name in ['syncCoLin_RandomUser', 'AsyncCoLin_RandomUser', 'AsyncCoLin_SelectUser', 'CoSingle', 'WCoLinUCB', 'WknowTheta', 'W_W0']:
                ThetaDiffList[alg_name] = []
            if alg_name in ['WCoLinUCB', 'WknowTheta', 'W_W0']:
                WDiffList[alg_name] = []

            for i in range(userSize):
                AccRegret[alg_name][i] = []
        
        
        # Loop begin
        for iter_ in range(self.iterations):
            # prepare to record theta estimation error
            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiff[alg_name] = 0
                if alg.CanEstimateCoUserPreference:
                    CoThetaDiff[alg_name] = 0
                if alg.CanEstimateW:
                    WDiff[alg_name] = 0
            self.regulateArticlePool(iter_) # ranomly generate article pool or regularly generate article pool 

            #noise = self.noise()
            noise = 0   # get rid of randomness from noise
            
            RandomUser = choice(self.users)            
            for alg_name, alg in algorithms.items():
                if 'SelectUser' in alg_name:
                    pickedUser, pickedArticle = alg.decide(self.articlePool, self.users)
                elif 'RandomUser' in alg_name:
                    pickedUser = RandomUser
                    pickedArticle = alg.decide(self.articlePool, pickedUser.id)
                    
                reward = self.getReward(pickedUser, pickedArticle) + noise

                #get optimal reward from chosen user
                #OptimalReward = self.GetOptimalReward(pickedUser, self.articlePool)  

                #get optimal reward from the best user+article combinations  
                OptimalUser, OptimalArticle, OptimalUserReward = self.GetOptimalUserReward(self.users, self.articlePool) 
                OptimalReward = OptimalUserReward + noise

                alg.updateParameters(pickedArticle, reward, pickedUser.id)

                regret = OptimalReward - reward    
                AccRegret[alg_name][pickedUser.id].append(regret)
                
                # Record parameter estimation error of all users
                for u in self.users:  
                    if alg.CanEstimateUserPreference:
                        ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))
                    if alg.CanEstimateCoUserPreference:
                        CoThetaDiff[alg_name] += self.getL2Diff(u.CoTheta, alg.getCoTheta(u.id))
                    if alg.CanEstimateW:
                        WDiff[alg_name] += self.getL2Diff(self.W.T[u.id], alg.getW(u.id))  

            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiffList[alg_name] += [ThetaDiff[alg_name]/userSize]
                if alg.CanEstimateCoUserPreference:
                    CoThetaDiffList[alg_name] += [CoThetaDiff[alg_name]/userSize]
                if alg.CanEstimateW:
                    WDiffList[alg_name] += [WDiff[alg_name]/userSize]
                
            if iter_%self.batchSize == 0:
                self.batchRecord(iter_)
                tim_.append(iter_)
                for alg_name in algorithms.iterkeys():
                    TotalAccRegret = sum(sum (u) for u in AccRegret[alg_name].itervalues())
                    BatchAverageRegret[alg_name].append(TotalAccRegret)

                # Save result into files if necessary
                '''
                with open(filenameWriteRegret, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(BatchAverageRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
                    f.write('\n')
                with open(filenameWritePara, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(CoThetaDiffList[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
                    f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.iterkeys()]))
                    f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in WDiffList.iterkeys()]))
                    f.write('\n')
                '''    
        # plot the results        
        f, axa = plt.subplots(2, sharex=True)
        for alg_name in algorithms.iterkeys():    
            axa[0].plot(tim_, BatchAverageRegret[alg_name],label = alg_name)

            #plt.lines[-1].set_linewidth(1.5)
            print '%s: %.2f' % (alg_name, BatchAverageRegret[alg_name][-1])
        axa[0].legend(loc='upper right',prop={'size':9})
        axa[0].set_xlabel("Iteration")
        axa[0].set_ylabel("Regret")
        axa[0].set_title("Accumulated Regret")
        
        # plot the estimation error of co-theta and theta
        time = range(self.iterations) 
        for alg_name, alg in algorithms.items():
            if alg.CanEstimateUserPreference:
                axa[1].plot(time, ThetaDiffList[alg_name], label = alg_name + '_Theta')
            if alg.CanEstimateCoUserPreference:
                axa[1].plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
        '''  
        for alg_name in algorithms.iterkeys():
            axa[1].plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
            # CoLin algorithm can estimate theta
            if alg_name == 'AsyncCoLin_RandomUser' or alg_name == 'AsyncCoLin_SelectUser':
                axa[1].plot(time, ThetaDiffList[alg_name], label = alg_name + '_Theta')
        '''        
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
    
    userFilename = path.join(sim_files_folder, "users_"+str(n_users)+"+dim-"+str(dimension)+ "Ugroups" + str(UserGroups)+".json")
    
    #"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
    # we can choose to simulate users every time we run the program or simulate users once, save it to 'sim_files_folder', and keep using it.
    UM = UserManager(dimension, n_users, UserGroups = UserGroups, thetaFunc=featureUniform, argv={'l2_limit':1})
    #users = UM.simulateThetafromUsers()
    #UM.saveUsers(users, userFilename, force = False)
    users = UM.loadUsers(userFilename)

    articlesFilename = path.join(sim_files_folder, "articles_"+str(n_articles)+"+dim"+str(dimension) + "Agroups" + str(ArticleGroups)+".json")
    # Similarly, we can choose to simulate articles every time we run the program or simulate articles once, save it to 'sim_files_folder', and keep using it.
    AM = ArticleManager(dimension, n_articles=n_articles, ArticleGroups = ArticleGroups, FeatureFunc=featureUniform,  argv={'l2_limit':1})
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

    simExperiment_SelectUser = simulateOnlineData_SelectUser(dimension  = dimension,
                        iterations = iterations,
                        articles=articles,
                        users = users,        
                        noise = lambda : np.random.normal(scale = NoiseScale),
                        batchSize = batchSize,
                        type_ = "UniformTheta", 
                        signature = AM.signature,
                        poolArticleSize = poolSize, NoiseScale = NoiseScale, epsilon = epsilon, Gepsilon =Gepsilon)

    selectUser_Algorithms= {}
    
    selectUser_Algorithms['LinUCB_SelectUser'] = LinUCB_SelectUserAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)    
    selectUser_Algorithms['AsyncCoLin_SelectUser'] = CoLinUCB_SelectUserAlgorithm(dimension=dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
    
    #selectUser_Algorithms['GOBUCB_SelectUser'] = GOBLin_SelectUserAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getGW())
    #selectUser_Algorithms['AsyncCoLin_RandomUser'] = AsyCoLinUCBAlgorithm(dimension=dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
    
    simExperiment_SelectUser.runAlgorithms(selectUser_Algorithms)