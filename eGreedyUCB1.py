from random import choice, random
import numpy as np

class ArticleBaseStruct(object):
    def __init__(self, articleID):
        self.articleID = articleID
        self.totalReward = 0.0
        self.numPlayed = 0
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1


class UCB1Struct(ArticleBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return 0
        else:
            return self.totalReward / self.numPlayed + np.sqrt(2*np.log(allNumPlayed) / self.numPlayed)
        
class eGreedyArticleStruct(ArticleBaseStruct):
    def getProb(self):
        if self.numPlayed == 0:
            pta = 0
        else:
            pta = self.totalReward/self.numPlayed
        return pta

class UCB1Algorithm:
    def __init__(self):
        self.articles = {}
        self.TotalPlayCounter = 0
        
    def decide(self, pool_articles, userID):
        self.TotalPlayCounter +=1
        article_Picked = None
        maxPTA = float('-inf')
        
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = UCB1Struct(x.id)

            if self.articles[x.id].numPlayed == 0:
                article_Picked = x
                return article_Picked

            x_pta = self.articles[x.id].getProb(self.TotalPlayCounter)
            if maxPTA < x_pta:
                article_Picked = x
                maxPTA = x_pta
        return article_Picked       
         
    def updateParameters(self, pickedArticle, click, userID):  #parameters: (pickedArticle, click)
        self.articles[pickedArticle.id].updateParameters(click)


class eGreedyAlgorithm(UCB1Algorithm):
    def __init__(self, epsilon):
        UCB1Algorithm.__init__(self)
        self.epsilon = epsilon

    def decide(self, pool_articles, userID):
        article_Picked = None
        if random() < self.epsilon: # random exploration
            article_Picked = choice(pool_articles)
            if article_Picked not in self.articles:
                self.articles[article_Picked.id] = eGreedyArticleStruct(article_Picked.id)
        else:
            maxPTA = float('-inf')
            for x in pool_articles:
                if x.id not in self.articles:
                    self.articles[x.id] = eGreedyArticleStruct(x.id)
                x_pta = self.articles[x.id].getProb()
                if maxPTA < x_pta:
                    article_Picked = x
                    maxPTA = x_pta
        return article_Picked