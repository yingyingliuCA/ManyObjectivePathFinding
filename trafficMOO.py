import os
import csv
import pandas as pd # csv handler
import networkx as nx # graph package
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt #uncomment for plots
import time
from platypus import NSGAII, DTLZ2, GeneticAlgorithm, Problem, Permutation, unique, ParticleSwarm, Binary, Constraint, RandomGenerator, TournamentSelector, PCX
import math
import datetime
from datetime import timedelta
import random
import copy

#references of NSGAII implmentation:
# 1. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II, Kalyanmoy Deb, Associate Member, IEEE, Amrit Pratap, Sameer Agarwal, and T. Meyarivan
# 2. A Nondominated Sorting Genetic Algorithm for Shortest Path Routing Problem, C. Chitra and P. Subbaraj
# 3. Multiobjective Environmentally Sustainable Road Network Design Using Pareto Optimization, Yi Wang and W. Y. Szeto
# 4. Code author: David Hadka, Accessed on July 10, 2019
# https://platypus.readthedocs.io/en/latest/
# 5. Code author: Michael Allen, Accessed on July 23, 2019
# https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/
# 6. The road network is from http://iot.ee.surrey.ac.uk:8080/datasets.html

#constants 
FIRST_NON_DOMINATED = 1
SECOND_NON_DOMINATED = 2
NO_DOMINATION = 0
CROSSOVER_NBX = 0 #Node based crossover
CROSSOVER_MNX = 1 #Multiple node crossover
CROSSOVER_PMX = 2 #Partially mapped crossover. Not implemented. PMX is crossover for permutations so it's not fit here
MUTATION_NO = 0 #No mutation
MUTATION_YES = 1 #Perform mutation

def is_valid_path(path, start, end):
	if path[0] == start and path[-1] == end:
		return True 
	else:
		return False

def generate_random_path(start, end):
	# create a loop free random path from global variable startNode to endNode on global graph G
	path = []
	nextNode = start 
	while nextNode != end:
		path.append(nextNode)
		neighbors = list(G.successors(nextNode))
		# print('path:',path)
		# print('neighbors:',neighbors)
		if neighbors == []: #this will be an invalid path
			return path
		else:		
			unvisitedNeighbors = []
			for node in neighbors: 
				if node not in path:
					unvisitedNeighbors.append(node)	
			if unvisitedNeighbors == []:
				return path #this will be an invalid path
			else:
				nextNode = random.choice(unvisitedNeighbors) #avoid loops
	path.append(nextNode) #append the end node
	return path #this will be a valid path	

def generate_valid_rand_path(start, end):
	randPath = generate_random_path(start, end)
	numInvalidPaths = 0
	while not is_valid_path(randPath, start, end):
		numInvalidPaths = numInvalidPaths + 1	
		# print('randPath:\n',randPath)
		randPath = generate_random_path(start, end)
	# print('numInvalidPaths:',numInvalidPaths)
	# print('valid path:',randPath)
	return randPath

def create_population():
	# create initPopulationSize(global variable) loop free valid random paths from global variable startNode to endNode on global graph G 
	# initialize population as an empty list 
	population = []
	for i in range(initPopulationSize):
		randPath = generate_valid_rand_path(startNode, endNode)
		population.append(randPath)
		# print('my path:',randPath)
	return population

def fitness_scores(path):
	total_distance = 0
	total_turns = 0	
	total_travelTime = 0
	total_emission = 0
	index = 0
	currNode = path[index]
	edges = list(G.edges())
	while currNode != endNode: #assume the path is valid (the last node is endNode)
		index = index + 1
		nextNode = path[index]
		total_distance = total_distance + G.edges[currNode, nextNode]['length']
		total_turns = total_turns + G.edges[currNode, nextNode]['isTurn']
		edgeId = edges.index((currNode, nextNode))
		total_travelTime = total_travelTime + travelTimeMatrix[edgeId][0] #static for now, change to time index for dynamic 
		total_emission = total_emission + emissionMatrix[edgeId][0]
		currNode = nextNode

	#convert distance from meters to KM and travel time from seconds to minutes
	return np.asarray([total_distance/1000.0, total_turns, total_travelTime/60.0, total_emission])
	# return [total_distance, total_turns, total_travelTime, total_emission]

def score_population(population):
	populationScores = np.zeros((len(population),numObjectives)) #4 objectives
	for i in range(len(population)):
		populationScores[i,:] = fitness_scores(population[i])
	return populationScores

def non_dominated(scores1, scores2): #arguments need to be in the form of array instead of list?
	if all(scores1 <= scores2) and any(scores1 < scores2): #minimization problem
		return FIRST_NON_DOMINATED #score 1 dominates score 2
	else:
		if all(scores2 <= scores1) and any(scores2 < scores1):
			return SECOND_NON_DOMINATED #score 2 dominates score 1
		else:
			return NO_DOMINATION #neither dominates the other		

def tournament_selection(population):
	firstFlag = 0 
	for i in range(tournamentSize):
		index = random.randint(0, len(population)-1)
		if firstFlag == 0:
			winner = index
			firstFlag = 1
		else:
			referencePath = population[winner]
			referenceScores = fitness_scores(referencePath)
			path = population[index]
			scores = fitness_scores(path)
			non_dominatedFlag = non_dominated(scores, referenceScores)
			if non_dominatedFlag == FIRST_NON_DOMINATED: #if scores dominates referenceScores
				winner = index
	return winner

def remove_loops(path):
	loopFreePath = copy.deepcopy(path)
	currIdx = 0
	while currIdx < len(loopFreePath)-1: #search until the second last node. while makes sure multiple loops are removed
		#search for duplicates
		currNode = loopFreePath[currIdx]
		# print('currIdx:',currIdx,',currNode:',currNode)
		while currNode in loopFreePath[currIdx+1:]:
			dupIdx = loopFreePath[currIdx+1:].index(currNode) + (currIdx+1) # e.g. [a, b, d, b, c], currIdx = 1, partial path = [b, c], dupIdx = 2 
			del loopFreePath[currIdx+1:dupIdx+1]
			# print (loopFreePath)
		currIdx = loopFreePath.index(currNode)+1		
	return loopFreePath

def mutation(population, mutationProbability):
	# print('population before mutation\n',population)
	for chromosome in population:
	#find a random node for mutation
		mutationIndex = random.randint(1, len(chromosome)-2) #do not mutation start and end nodes
		randNum = random.uniform(0,1)
		if randNum < mutationProbability:			
			subpath1 = chromosome[0:mutationIndex]
			prevNode = chromosome[mutationIndex-1]
			candidateNodes = list(G.successors(prevNode))
			mutationNode = random.choice(candidateNodes)
			subpath2 = generate_valid_rand_path(mutationNode, endNode)
			chromosome = subpath1 + subpath2
			chromosome = remove_loops(chromosome)
	# print('population after mutation\n',population)
	return population


def crossover_NBX(population, maxNumCrossoverPoints = 1):
	parent1 = population[tournament_selection(population)]
	parent2 = population[tournament_selection(population)]
	# print('parent1:\n',parent1)
	# print('parent2:\n',parent2)
	#find a random node that exists in both parents. The node is neither startNode nor endNode
	commonNodes = list((set(parent1).intersection(set(parent2))).difference({startNode,endNode}))
	# print('commonNodes:\n',commonNodes)
	while len(commonNodes) == 0: #if no common node except for start and end nodes, select different parents
		parent1 = population[tournament_selection(population)]
		parent2 = population[tournament_selection(population)]
		commonNodes = list((set(parent1).intersection(set(parent2))).difference({startNode,endNode}))
	numCrossOvers = min(len(commonNodes), maxNumCrossoverPoints)
	crossOverPoints = []
	for i in range(numCrossOvers):
		index = random.randint(0, len(commonNodes)-1) 
		crossOverPoints.append(commonNodes[index])
	#remove duplicates if any. Chance is small
	crossOverPoints = list(dict.fromkeys(crossOverPoints))

	#crossover
	child1=[]
	child2=[]
	#there is at least 1 crossover point
	parent1CrossOverIdx = parent1.index(crossOverPoints[0])
	parent2CrossOverIdx = parent2.index(crossOverPoints[0])
	child1 = child1+parent1[0:parent1CrossOverIdx]
	child2 = child2+parent2[0:parent2CrossOverIdx]
	for i in range(1, len(crossOverPoints)):
		parent1NextCrossOverIdx = parent1.index(crossOverPoints[i])
		parent2NextCrossOverIdx = parent2.index(crossOverPoints[i])
		child1 = child1 + parent2[parent2CrossOverIdx:parent2NextCrossOverIdx]
		child2 = child2 + parent1[parent1CrossOverIdx:parent1NextCrossOverIdx]
		parent1CrossOverIdx = parent1NextCrossOverIdx
		parent2CrossOverIdx = parent2NextCrossOverIdx
	child1 = child1 + parent2[parent2CrossOverIdx:]
	child2 = child2 + parent1[parent1CrossOverIdx:]
	# print('parent1:',parent1)
	# print('parent2:',parent2)
	# print('crossOverPoints:',crossOverPoints)
	# print('child1 before removing loops:',child1)
	# print('child2 before removing loops:',child2)
	child1 = remove_loops(child1)
	child2 = remove_loops(child2)
	# print('child1 after removing loops:',child1)
	# print('child2 after removing loops:',child2)
	# print('crossOverNode:',crossOverNode,',parent1CrossOverIdx:',parent1CrossOverIdx,',parent2CrossOverIdx:',parent2CrossOverIdx)
	# child1 = remove_loops(parent1[0:parent1CrossOverIdx]+parent2[parent2CrossOverIdx:])
	# child2 = remove_loops(parent2[0:parent2CrossOverIdx]+parent1[parent1CrossOverIdx:])
	# print('parent1[0:parent1CrossOverIdx]:',parent1[0:parent1CrossOverIdx])
	# print('parent2[parent2CrossOverIdx:]:',parent2[parent2CrossOverIdx:])
	return child1, child2
#todo: multiple point crossover and PMX 
#benefit of PMX: don't have to have common node - but it is for permutation where two parents are aligned
 
def breed_population(population, crossoverFlag = CROSSOVER_NBX, mutationFlag = MUTATION_NO, mutationProbability = 0.5): #defaults
	newPopulation = []
	for i in range(int(len(population)/2)):
		if crossoverFlag == CROSSOVER_NBX:
			child1,child2 = crossover_NBX(population, 1)
		if crossoverFlag == CROSSOVER_MNX:
			child1,child2 = crossover_NBX(population, 5) #maximum 5 crossover nodes
		newPopulation.append(child1)
		newPopulation.append(child2)
	#allow parents and children to compete
	population = population + newPopulation
	#mutation
	if mutationFlag == MUTATION_YES:
		population = mutation(population, mutationProbability)
	# #allow parents and children to compete
	# return population + newPopulation
	return population

def identify_pareto(scores, populationIds):
	populationSize  = scores.shape[0]
	#start off being labelled as on the pareto front
	paretoFront = np.ones(populationSize, dtype=bool)
	for i in range(populationSize):
		for j in range(populationSize):
			if i != j: 
				nonDominatedFlag = non_dominated(scores[i], scores[j])
				if nonDominatedFlag == SECOND_NON_DOMINATED:
					paretoFront[i] = 0
					#stop further comparisons with i
					break 
	return populationIds[paretoFront]

def calculate_crowding(scores):
	populationSize = scores.shape[0]
	numberOfScores = scores.shape[1]
	crowdingMatrix = np.zeros((populationSize, numberOfScores))
	# print('calculate_crowding scores:\n',scores,'\nscores.min(0):',scores.min(0),',scores.max(0),',scores.max(0), 'scores.min(1):',scores.min(1))
	#min(0): minimums in the columns
	#min(1): minimums in the rows
	#nuympy.ptp: maximum - minimum.
	#It doesn't matter whether this is maximization or minimization problem. crowd distance is not about how
	#close the solution is to the optimal, it is about how close the solultion is to its neighbours. 
	#crowd distance is used to preserve diversity
	normedScores = (scores - scores.min(0))/ scores.ptp(0) 
	for col in range(numberOfScores):
		crowding = np.zeros(populationSize)
		# end points (minimum and maximum objective values) have maximum crowding
		crowding[0] = 1 
		crowding[populationSize-1] = 1
		# sort each score
		sortedScores = np.sort(normedScores[:,col])
		sortedScoresIndex = np.argsort(normedScores[:,col])
		# calculate crowding distance for each individual 
		# crowding between adjacent scores
		# print('sortedScores:\n',sortedScores)
		# print('sortedScoresIndex:\n',sortedScoresIndex)
		crowding[1:populationSize-1] = \
		(sortedScores[2:populationSize] - sortedScores[0: populationSize-2])
		# print('crowding:\n',crowding)
		# resort to original order 
		reSortOrder = np.argsort(sortedScoresIndex)
		sortedCrowding = crowding[reSortOrder]
		# print('reSortOrder:\n',reSortOrder)
		# print('sortedCrowding:\n',sortedCrowding)
		crowdingMatrix[:,col] = sortedCrowding
	crowdingDistances = np.sum(crowdingMatrix, axis = 1)
	# print('crowdingDistances:\n', crowdingDistances)
	return crowdingDistances

def crowding_selection(scores, numberToSelect):
	#all the scores come from the same pareto front
	#this is because in the calling function, we only need to call reduce_by_crowding if
	#1) previous pareto fronts do not meet population size so a new pareto front is needed, and
	#2) after calculating the new pareto front, the combined size exceeeds population size
	# because lower ranked (better) front is always preferred than the worse front, we only reduce the population
	# from the worse front. Use tournament selection to pick two chromosomes, and choose the one with larger crowding distance
	populationIds = np.arange(scores.shape[0])
	crowdingDistances = calculate_crowding(scores)
	unselectedPopulationIdsList = list(populationIds)
	selectedPopulationIdsList = []
	for i in range(numberToSelect):
		candidate1Id = random.choice(unselectedPopulationIdsList)
		candidate2Id = random.choice(unselectedPopulationIdsList)
		#by default, set winner to the first candidate
		winner = candidate1Id
		#winner is second candidate if it has larger crowding distance
		if crowdingDistances[candidate2Id] >= crowdingDistances[candidate1Id]:
			winner = candidate2Id
		selectedPopulationIdsList.append(winner)
		unselectedPopulationIdsList.remove(winner)
	return np.asarray(selectedPopulationIdsList)


def build_pareto_population(population, scores):
	allPopulationIds = np.arange(len(population))
	# print('allPopulationIds:\n',allPopulationIds)
	unselectedPopulationIds = allPopulationIds
	paretoFront = []
	while len(paretoFront) < initPopulationSize: 
		# print('Identify Pareto Front...\n')
		# Identify the first Pareto front (F1)
		tempParetoFront = identify_pareto(scores[unselectedPopulationIds, :], unselectedPopulationIds)
		# if the size of solution sets (F1, F2, ...) is larger than the maximum permitted solution then reduce the size of the set by crowding selection
		combinedParetoSize = len(paretoFront) + len(tempParetoFront)
		if combinedParetoSize > initPopulationSize:
			# print('before crowding selection, tempParetoFront is:\n',tempParetoFront)
			# numberToReduce = combinedParetoSize - initPopulationSize
			numberToSelect = initPopulationSize - len(paretoFront)
			#crowd selection the solutions that preserve diversity 
			selectedPopulationIds = crowding_selection(scores[tempParetoFront], numberToSelect)
			tempParetoFront = tempParetoFront[selectedPopulationIds];
			# print('after crowding selection, tempParetoFront is:\n',tempParetoFront)
		# combine pareto fronts
		paretoFront = np.hstack((paretoFront,tempParetoFront))
		# if F1 is smaller than required population size then repeat pareto selection (after removal of selected). This new set of solutions is F2
		unselectedSet = set(allPopulationIds)-set(paretoFront)
		unselectedPopulationIds = np.array(list(unselectedSet))
		# print('Pareto front at this iteration: len=',len(paretoFront), '\nIds:\n',paretoFront)
	newPopulation = [] #new population should have initPopulationSizes
	paretoFrontList = list(paretoFront)
	for i in range(len(population)):
		if i in paretoFrontList:
			newPopulation.append(population[i])
	# print('new population size:',len(newPopulation))
	return newPopulation

class NSGAIICustom(NSGAII):
	def __init__(self, problem,
	         population_size = 100,
	         generator = RandomGenerator(),
	         selector = TournamentSelector(2),
	         variator = None,
	         archive = None,
	         **kwargs):
		super(NSGAIICustom, self).__init__(problem, population_size, generator, **kwargs)
		self.selector = selector
		self.variator = variator
		self.archive = archive

  #   def initialize(self):
  #   	self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
		# print('my population:',self.population)
		# self.evaluate_all(self.population)
		# if self.archive is not None:
		#     self.archive += self.population       
		# if self.variator is None:
		#     self.variator = default_variator(self.problem)       	

# function author: Ziyue Wang
# modified by Ying Ying Liu
def process_static(debug = False):
	G = nx.DiGraph()
	metaData = pd.read_csv('data/raw/trafficMetaData.csv')
	# G: direct network, distance (length) on edge
	# find all nodes
	nodes = dict()
	for i in range(len(metaData['POINT_1_STREET'])):
		nodes[metaData['POINT_1_NAME'][i]] = (float(metaData['POINT_1_LAT'][i]), 
			float(metaData['POINT_1_LNG'][i]))
		nodes[metaData['POINT_2_NAME'][i]] = (float(metaData['POINT_2_LAT'][i]), 
			float(metaData['POINT_2_LNG'][i]))
	# insert nodes to the graph
	# uniqueNodes = dict()
	# i = 0
	for node in nodes:
		# if node not in uniqueNodes: 
		# 	uniqueNodes[node] = i
		G.add_node(str(node),pos=(nodes[node][0],nodes[node][1]))
		# G.add_node(str(node),pos=(nodes[node][0],nodes[node][1]), nodeId = i)
		# i = i + 1
	# add edges
	# uniqueEdges = dict()
	# j = 0
	for i in range(len(metaData['POINT_1_STREET'])):
		s = metaData['POINT_1_NAME'][i]
		t = metaData['POINT_2_NAME'][i]
		l = int(metaData['DISTANCE_IN_METERS'][i])
		speedLimit = int(metaData['NDT_IN_KMH'][i])
		isTurn = 0
		sStreet = str(metaData['POINT_1_STREET'][i]).strip()
		tStreet = str(metaData['POINT_2_STREET'][i]).strip()
		if sStreet != tStreet:
			isTurn = 1
		# c = int(random()*10) #YY: static randome time between 1 and 10 minutes
		rep = metaData['REPORT_ID'][i]
		# if (s,t) not in uniqueEdges and (t,s) not in uniqueEdges:
		# 	uniqueEdges[s,t] = j
		# 	uniqueEdges[t,s] = j
		# G.add_edge(str(s), str(t), length = str(l), report = str(rep))
		#todo: one edge has two report ID's. For example, 4364 to 4349 has report 158895 and 173118
		G.add_edge(str(s), str(t), length = l, report = str(rep), isTurn = isTurn, speedLimit = speedLimit) #YY
		#G.add_edge(str(s), str(t), length = l, report = str(rep), edgeId = uniqueEdges[s,t]) #YY
		#j = j + 1
	nx.write_gml(G, 'data/citypulse_static.gml')
	# print('uniqueEdges:',len(uniqueEdges),',uniqueNodes:', len(uniqueNodes),'G.number_of_edges()=',G.number_of_edges(),'G.number_of_nodes()=',G.number_of_nodes())
	return G

def read_traffic_data():
	G = process_static()
	# G2 = nx.read_gml('data/citypulse_static.gml')
	##uncomment to draw graph
	# pos=nx.spring_layout(G)
	# lab=dict(zip(G,G.nodes())) #use node id as label
	# nx.draw(G,pos=pos,node_size=15,with_labels=False) #first suppress the default labels
	# nx.draw_networkx_labels(G,pos, lab, font_size=6) #then draw with the custom labels
	# plt.show()
	# print('Done.')
	#first read the minimum and maximum timestamp to find the size of the dynamic data
	minTimeStamp = '2014-02-13T11:30:00'
	maxTimeStamp = '2014-06-09T05:35:00'
	for file in os.listdir('data/raw/traffic_feb_june/'):
		data = pd.read_csv('data/raw/traffic_feb_june/'+file)
		if minTimeStamp > min(data['TIMESTAMP']):
			minTimeStamp = min(data['TIMESTAMP'])
		if maxTimeStamp < max(data['TIMESTAMP']):
			maxTimeStamp = max(data['TIMESTAMP'])
	minTimeStampObj = datetime.datetime.strptime(minTimeStamp,'%Y-%m-%dT%H:%M:%S')
	maxTimeStampObj = datetime.datetime.strptime(maxTimeStamp,'%Y-%m-%dT%H:%M:%S')
	numTimeStamps = int((maxTimeStampObj - minTimeStampObj)/timedelta(minutes=5))+1
	distMatrix = np.full((G.number_of_nodes(),G.number_of_nodes()),math.inf)
	turnMatrix = np.full((G.number_of_nodes(),G.number_of_nodes()),math.inf)
	travelTimeMatrix = np.full((G.number_of_edges(),numTimeStamps), math.inf)
	emissionMatrix = np.full((G.number_of_edges(),numTimeStamps), math.inf)
	edges = list(G.edges)
	nodes = list(G.nodes)
	for i in range(G.number_of_edges()):
		s = edges[i][0]
		t = edges[i][1]
		length = G.edges[s,t]['length']
		speedLimit = G.edges[s,t]['speedLimit']
		sNodeId = nodes.index(s)
		tNodeId = nodes.index(t)
		edgeId = i
		rpt = G.edges[s,t]['report']
		distMatrix[sNodeId][tNodeId] = length #in meters
		turnMatrix[sNodeId][tNodeId] = G.edges[s,t]['isTurn']
		trafficDataPath = 'data/raw/traffic_feb_june/trafficData'+str(rpt)+'.csv'
		trafficData = pd.read_csv(trafficDataPath)
		rowIdx = 0
		timeIdx = 0
		lengthInFoot = length * 3.28084 # 1 meter = 3.28084 foot
		while timeIdx < 12: #just read 12 instances for quick testing
			posTimeObj = minTimeStampObj + timeIdx*timedelta(minutes=5)
			currTimeObj = datetime.datetime.strptime(trafficData['TIMESTAMP'][rowIdx],'%Y-%m-%dT%H:%M:%S')
			if currTimeObj - posTimeObj == timedelta(0):
				avgSpeed = trafficData['avgSpeed'][rowIdx]
				vehicleCount = trafficData['vehicleCount'][rowIdx]
				avgSpeedMeterPerSec = avgSpeed * 0.44704 #1 mile per hour = 0.44704 meter per second
				speedLimitMeterPerSec = speedLimit * 0.44704
				if avgSpeedMeterPerSec > 0:
					travelTimeMatrix[edgeId][timeIdx] = length / avgSpeedMeterPerSec
				else:
					travelTimeMatrix[edgeId][timeIdx] = length / speedLimitMeterPerSec
				#emission Wang 2017 CO2 - equation 9 and table 1
				avgSpeedFootPerSecond = avgSpeed * 1.46667 #1 mile per hour = 1.46667 foot per second
				if trafficData['avgMeasuredTime'][rowIdx] > 0:
					vehiclePerSecond = vehicleCount*1.0 / trafficData['avgMeasuredTime'][rowIdx] 				
				else:
					vehiclePerSecond = 0
				vehiclePerHour = vehiclePerSecond * 3600
				if avgSpeedFootPerSecond > 0:
					speedForTEC = avgSpeedFootPerSecond
				else:
					speedForTEC = speedLimitMeterPerSec
				TEC = (0.00051*3.3963*math.exp(0.014561*speedForTEC)/(1000*speedForTEC))*lengthInFoot*vehiclePerHour
				+ (0.00136*2.7843*math.exp(0.015062*speedForTEC)/(10000*speedForTEC))*lengthInFoot*vehiclePerHour
				+ (0.00103*1.5718*math.exp(0.040732*speedForTEC)/(10000*speedForTEC))*lengthInFoot*vehiclePerHour				
				emissionMatrix[edgeId][timeIdx] = TEC 
				rowIdx = rowIdx + 1
			timeIdx = timeIdx + 1
	print('data read complete')
	s0 = edges[0][0]
	t0 = edges[0][1]
	rpt0 = G.edges[s0,t0]['report']
	return G, distMatrix, turnMatrix, travelTimeMatrix, emissionMatrix

def MOOSP(x): #for Platypus package (not good for path finding)
	selection = x[0]
	total_distance = 0
	total_turns = 0	
	total_travelTime = 0
	total_emission = 0 

	# the following finds random path generated by NSGAII
	for i in range(problemSize):
		if selection[i]: #if true
			sIdx = int(i / numNodes)
			tIdx = int(i % numNodes)
			s = nodes[sIdx]
			t = nodes[tIdx]
			# print('s=',s,',t=',t)
			# if distMatrix[sIdx][tIdx] != math.inf:
			if (s, t) in edges:
				edgeId = edges.index((s, t))
				if distMatrix[sIdx][tIdx] == math.inf:
					total_distance = math.inf
					total_turns = math.inf
					# print('edgeId:',edgeId, ',sIdx:',sIdx, ',tIdx:',tIdx, 'distMatrix[sIdx][tIdx]=',distMatrix[sIdx][tIdx], 'travelTimeMatrix[][]=',travelTimeMatrix[edgeId][0])
					break
				else:
					# rpt = G.edges[s,t]['report']
					# print('edgeId:',edgeId, ',sIdx:',sIdx, ',tIdx:',tIdx, ',s:',s,',t:',t,'report:',rpt,'distMatrix[sIdx][tIdx]=',distMatrix[sIdx][tIdx], 'travelTimeMatrix[][]=',travelTimeMatrix[edgeId][0])					
					total_distance = total_distance + int(distMatrix[sIdx][tIdx]) #?if do not use int(), will get this error: TypeError: object of type 'numpy.float64' has no len()
					total_turns = total_turns + int(turnMatrix[sIdx][tIdx])
				# total_distance = total_distance + distMatrix[sIdx][tIdx]				
				if travelTimeMatrix[edgeId][0] == math.inf:
					total_travelTime = math.inf 
					break
				else:
					total_travelTime = total_travelTime + float(travelTimeMatrix[edgeId][0])				
				if emissionMatrix[edgeId][0] == math.inf:
					total_emission = math.inf 
					break
				else:
					# print('edgeId:',edgeId,'emissionMatrix[][]=',emissionMatrix[edgeId][0])
					total_emission = total_emission + float(emissionMatrix[edgeId][0])					
	# print('my total_distance:',total_distance)
	# print('my total_turns:',total_turns)
	# print('my total_travelTime:',total_travelTime)	
	# print('my total_emission:',total_emission)	
	return total_distance, total_turns, total_travelTime, total_emission

#main program
#global variables
G, distMatrix, turnMatrix, travelTimeMatrix, emissionMatrix = read_traffic_data()
edges = list(G.edges)
nodes = list(G.nodes)
numNodes = G.number_of_nodes()
#single objective shortest path:
startNode = '4320'
endNode = '4551'
startIdx = nodes.index(startNode)
endIdx = nodes.index(endNode)
problemSize = numNodes * numNodes
astarPath = nx.astar_path(G, startNode,endNode, weight='length')
print('A* path:\n',astarPath)
print('A* path scores:\n',fitness_scores(astarPath))
initPopulationSize = 100
numGenerations = 50
tournamentSize = 2 
numObjectives = 4 
numConfigs = 4
population = create_population()
# print('population:\n',population)
for config in range(numConfigs): 
	print('NSGAII Configuration ',config)
	for generation in range(numGenerations):
		if generation % 10 == 0:
			print('Generation %i\n'%generation)
		if config == 0:
			population = breed_population(population) #default: NBX with no mutation 
		if config == 1: 
			population = breed_population(population, CROSSOVER_MNX, MUTATION_NO) #MNX with no mutation
		if config == 2:
			population = breed_population(population, CROSSOVER_MNX, MUTATION_YES, 0.5) #MNX with mutation prob 0.5
		if config == 3:
			population = breed_population(population, CROSSOVER_MNX, MUTATION_YES, 0.8) #MNX with mutation prob 0.8		
		scores = score_population(population)
		population = build_pareto_population(population, scores)
	#get final pareto front
	scores = score_population(population)
	populationIds = np.arange(len(population))
	print('before final pareto front: len of population:', len(population))
	paretoFront = identify_pareto(scores, populationIds)
	paretoFrontList = list(paretoFront)
	finalSolutions = []
	finalScores = []
	for i in range(len(population)):
		if i in paretoFrontList:
			finalSolutions.append(population[i])
			finalScores.append(scores[i])
	print('after final pareto front: len of population:', len(finalSolutions))
	print('final solutions:')
	for i in range(len(finalSolutions)):
		print('path ',i,':',finalSolutions[i])
		print('scores:',finalScores[i])

	finalScores = np.asarray(finalScores)

	fig = plt.figure()
	# fig.set_size_inches(20, 20)
	ax = fig.add_subplot(111, projection='3d')
	title = 'From Node 4320(Hinnerup) to Node 4551(Hasselager) in Aarhus, Denmark\n'
	if config == 0:
		title = title + 'NSGAII - Single Point Crossover, No Mutation'
	if config == 1:
		title = title + 'NSGAII - Multi Point Crossover, No Mutation'
	if config == 2:
		title = title + 'NSGAII - Multi Point Crossover, Mutation Prob 0.5'		
	if config == 3:
		title = title + 'NSGAII - Multi Point Crossover, Mutation Prob 0.8'					
	fig.suptitle(title)
	x = finalScores[:,0]
	y = finalScores[:,1]
	z = finalScores[:,2]
	c = finalScores[:,3]
	ax.set_xlabel('Objective 1 - min distance (KM)')
	ax.set_ylabel('Objective 2 - min turns')
	ax.set_zlabel('Objective 3 - min travel time (minutes)')
	# ax.set_xlim([20, 40])  #hard code these limits here for now for fair visual comparison - not flexible
	# ax.set_ylim([8, 32])
	# ax.set_zlim([15, 70])

	# img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
	pnt3d = ax.scatter(x, y, z, c=c)
	cbar = plt.colorbar(pnt3d)
	cbar.set_label("Objective 4 - min total vehicle emission cost (TEC)")

	# fig.colorbar(img)
	figName = 'MultObjectivePathFindingConfig'+str(config)+'.png'
	plt.savefig(figName)
	# plt.show()

print('done')




