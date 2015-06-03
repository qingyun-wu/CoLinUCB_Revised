import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
	def __init__(self, id, theta = None, CoTheta = None):
		self.id = id
		self.theta = theta
		self.CoTheta = CoTheta


class UserManager():
	def __init__(self, dimension, userNum,  UserGroups, thetaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.userNum = userNum
		self.UserGroups = UserGroups
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def saveUsers(self, users, filename, force = False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			for i in range(len(users)):
				print users[i].theta
				f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')
				
	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta = json.loads(line)
				users.append(User(id, np.array(theta)))
		return users

	def generateMasks(self):
		mask = {}
		for i in range(self.UserGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateThetafromUsers(self):
		usersids = {}
		users = []
		mask = self.generateMasks()

		for i in range(self.UserGroups):
			usersids[i] = range(self.userNum*i/5, (self.userNum*(i+1))/5)

			for key in usersids[i]:
				thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
				l2_norm = np.linalg.norm(thetaVector, ord =2)
				users.append(User(key, thetaVector/l2_norm))
		'''
		usersids_1 = range(numUsers/5)
		usersids_2 = range(numUsers/5, numUsers*2/5)
		usersids_3 = range(numUsers*2/5, numUsers*3/5)
		usersids_4 = range(numUsers*3/5, numUsers*4/5)
		usersids_5 = range(numUsers*4/5, numUsers*5/5)

		mask1 = [0,1,1,1,1]
		mask2 = [1,0,1,1,1]
		mask3 = [1,1,0,1,1]
		mask4 =[1,1,1,0,1]
		mask5 = [1,1,1,1,0]
		users = []

		for key in usersids_1:
			users.append(User(key, np.multiply(thetaFunc(self.dimension, argv = argv), mask1)))
		for key in usersids_2:
			users.append(User(key, np.multiply(thetaFunc(self.dimension, argv = argv), mask2)))
		for key in usersids_3:
			users.append(User(key, np.multiply(thetaFunc(self.dimension, argv = argv), mask3)))
		for key in usersids_4:
			users.append(User(key, np.multiply(thetaFunc(self.dimension, argv = argv), mask4)))
		for key in usersids_5:
			users.append(User(key, np.multiply(thetaFunc(self.dimension, argv = argv), mask5)))
		'''
		
		return users

