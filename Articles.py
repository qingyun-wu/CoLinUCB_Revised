import cPickle
import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample, randint
import json

class Article():	
	def __init__(self, aid, FV=None):
		self.id = aid
		self.featureVector = FV
		

class ArticleManager():
	def __init__(self, dimension, n_articles, ArticleGroups, FeatureFunc, argv ):
		self.signature = "Article manager for simulation study"
		self.dimension = dimension
		self.n_articles = n_articles
		self.ArticleGroups = ArticleGroups
		self.FeatureFunc = FeatureFunc
		self.argv = argv
		self.signature = "A-"+str(self.n_articles)+"+AG"+ str(self.ArticleGroups)+"+TF-"+self.FeatureFunc.__name__

	def saveArticles(self, Articles, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(Articles)):
				f.write(json.dumps((Articles[i].id, Articles[i].featureVector.tolist())) + '\n')


	def loadArticles(self, filename):
		articles = []
		with open(filename, 'r') as f:
			for line in f:
				aid, featureVector = json.loads(line)
				articles.append(Article(aid, np.array(featureVector)))
		return articles

	#automatically generate masks for articles, but it may generate same masks
	def generateMasks(self):
		mask = {}
		for i in range(self.ArticleGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateArticlePool(self):
		articles = []
		
		articles_id = {}
		mask = self.generateMasks()

		if self.ArticleGroups > 1:
			for i in range(self.ArticleGroups):
				articles_id[i] = range((self.n_articles*i)/self.ArticleGroups, (self.n_articles*(i+1))/self.ArticleGroups)

				for key in articles_id[i]:
					featureVector = np.multiply(featureUniform(self.dimension, {}), mask[i])
					l2_norm = np.linalg.norm(featureVector, ord =2)
					articles.append(Article(key, featureVector/l2_norm ))

		else:
			for i in range(self.n_articles):
				featureVector = featureUniform(self.dimension, {})
				l2_norm = np.linalg.norm(featureVector, ord =2)
				articles.append(Article(i, featureVector/l2_norm ))
	
		return articles

