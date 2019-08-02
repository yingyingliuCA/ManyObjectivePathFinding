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
import plotly.graph_objects as go
import plotly.express as px 
import plotly.io
import sys

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
DYNAMIC_NO = 0
DYNAMIC_YES = 1

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

def fitness_scores(path, dynamicFlag):
	total_distance = 0
	total_turns = 0	
	total_travelTime = 0
	total_emission = 0
	index = 0
	currNode = path[index]
	currTimeIdx = 0
	edges = list(G.edges())

	while currNode != endNode: #assume the path is valid (the last node is endNode)
		index = index + 1
		nextNode = path[index]
		if G.has_edge(currNode, nextNode): 
			total_distance = total_distance + G.edges[currNode, nextNode]['length']
			total_turns = total_turns + G.edges[currNode, nextNode]['isTurn']
			edgeId = edges.index((currNode, nextNode))
			if dynamicFlag == DYNAMIC_NO:
				total_travelTime = total_travelTime + travelTimeMatrix[edgeId][0] #static for now, change to time index for dynamic 
				total_emission = total_emission + emissionMatrix[edgeId][0]
				# print('total_travelTime:',total_travelTime,',total_emission:',total_emission)
			if dynamicFlag == DYNAMIC_YES:
				total_travelTime = total_travelTime + travelTimeMatrix[edgeId][currTimeIdx]
				# total_emission = total_emission + emissionMatrix[edgeId][currTimeIdx]
				total_emission = total_emission + emissionMatrix[edgeId][currTimeIdx]*travelTimeMatrix[edgeId][currTimeIdx]/3600 #TECPerHour*TravelTimeSecond/3600 sec per hour
				currTimeIdx = math.floor(total_travelTime/300) #every time instance is 5 minutes (300 seconds)
				# print('total_travelTime:',total_travelTime,',total_emission:',total_emission,',currTimeIdx:',currTimeIdx, ',travelTimeMatrix[edgeId][currTimeIdx]=',travelTimeMatrix[edgeId][currTimeIdx])
			currNode = nextNode
		else: #in any case if the edge is invalid (maybe through deleting loops), assign large numbers to the scores to make sure this solution is eliminated
			total_emission = sys.maxsize
			total_travelTime = sys.maxsize
			total_turns = sys.maxsize
			total_distance = sys.maxsize
			break 

	#convert distance from meters to KM and travel time from seconds to minutes
	# return np.asarray([total_distance/1000.0, total_turns, total_travelTime/60.0, total_emission])
	return np.asarray([total_emission, total_travelTime/60.0, total_turns, total_distance/1000.0])
	# return [total_distance, total_turns, total_travelTime, total_emission]

def score_population(population, dynamicFlag):
	populationScores = np.zeros((len(population),numObjectives)) #4 objectives
	for i in range(len(population)):
		populationScores[i,:] = fitness_scores(population[i], dynamicFlag)
	return populationScores

def non_dominated(scores1, scores2): #arguments need to be in the form of array instead of list?
	if all(scores1 <= scores2) and any(scores1 < scores2): #minimization problem
		return FIRST_NON_DOMINATED #score 1 dominates score 2
	else:
		if all(scores2 <= scores1) and any(scores2 < scores1):
			return SECOND_NON_DOMINATED #score 2 dominates score 1
		else:
			return NO_DOMINATION #neither dominates the other		

def tournament_selection(population, dynamicFlag):
	firstFlag = 0 
	for i in range(tournamentSize):
		index = random.randint(0, len(population)-1)
		if firstFlag == 0:
			winner = index
			firstFlag = 1
		else:
			referencePath = population[winner]
			referenceScores = fitness_scores(referencePath, dynamicFlag)
			path = population[index]
			scores = fitness_scores(path, dynamicFlag)
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


def crossover_NBX(population, dynamicFlag, maxNumCrossoverPoints = 1):
	parent1 = population[tournament_selection(population, dynamicFlag)]
	parent2 = population[tournament_selection(population, dynamicFlag)]
	# print('parent1:\n',parent1)
	# print('parent2:\n',parent2)
	#find a random node that exists in both parents. The node is neither startNode nor endNode
	commonNodes = list((set(parent1).intersection(set(parent2))).difference({startNode,endNode}))
	# print('commonNodes:\n',commonNodes)
	while len(commonNodes) == 0: #if no common node except for start and end nodes, select different parents
		parent1 = population[tournament_selection(population, dynamicFlag)]
		parent2 = population[tournament_selection(population, dynamicFlag)]
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
 
def breed_population(population, dynamicFlag, crossoverFlag = CROSSOVER_NBX, mutationFlag = MUTATION_NO, mutationProbability = 0.5): #defaults
	newPopulation = []
	for i in range(int(len(population)/2)):
		if crossoverFlag == CROSSOVER_NBX:
			child1,child2 = crossover_NBX(population, dynamicFlag, 1)
		if crossoverFlag == CROSSOVER_MNX:
			child1,child2 = crossover_NBX(population, dynamicFlag, 5) #maximum 5 crossover nodes
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
		speedLimit = int(metaData['NDT_IN_KMH'][i]) #not sure if NDT_IN_KMH is speed limit, but it can also be treated like average speed
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
	# pos={}
	# for node in G.nodes():
	# 	pos[node] = [G.node[node]['pos'][0],G.node[node]['pos'][1]]
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
		speedLimit = G.edges[s,t]['speedLimit'] #KMH
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
		# while timeIdx < 12: #just read 12 instances for quick testing
		while timeIdx < 100:
			posTimeObj = minTimeStampObj + timeIdx*timedelta(minutes=5)
			currTimeObj = datetime.datetime.strptime(trafficData['TIMESTAMP'][rowIdx],'%Y-%m-%dT%H:%M:%S')
			if currTimeObj - posTimeObj == timedelta(0):
				avgSpeed = trafficData['avgSpeed'][rowIdx] #average speed should be km/h
				vehicleCount = trafficData['vehicleCount'][rowIdx]
				avgSpeedMeterPerSec = avgSpeed * 0.277778 #1 km per hour = 0.277778 meter per second
				speedLimitMeterPerSec = speedLimit * 0.277778
				if avgSpeedMeterPerSec > 0:
					travelTimeMatrix[edgeId][timeIdx] = length / avgSpeedMeterPerSec
				else:
					travelTimeMatrix[edgeId][timeIdx] = length / speedLimitMeterPerSec
				#emission Wang 2017 CO2 - equation 9 and table 1
				avgSpeedFootPerSecond = avgSpeed * 0.911344 #1 km per hour = 0.911344 foot per second
				speedLimitFootPerSecond = speedLimit * 0.911344
				if trafficData['avgMeasuredTime'][rowIdx] > 0:
					vehiclePerSecond = float(vehicleCount+1) / trafficData['avgMeasuredTime'][rowIdx]  #count the current vehicle				
				else:
					# vehiclePerSecond = 0
					vehiclePerSecond = float(1.0)/300   #count the current vehicle only, and divide by the default 5 minutes interval
					# need to avoid vehicle 
				vehiclePerHour = vehiclePerSecond * 3600
				if avgSpeedFootPerSecond > 0:
					speedForTEC = avgSpeedFootPerSecond
				else:
					speedForTEC = speedLimitFootPerSecond
				#emission: 
				# TEC =
				TECPerHour = (0.00051*3.3963*math.exp(0.014561*speedForTEC)/(1000*speedForTEC))*lengthInFoot*vehiclePerHour
				+ (0.00136*2.7843*math.exp(0.015062*speedForTEC)/(10000*speedForTEC))*lengthInFoot*vehiclePerHour
				+ (0.00103*1.5718*math.exp(0.040732*speedForTEC)/(10000*speedForTEC))*lengthInFoot*vehiclePerHour							
				#debug: 
				# if (edgeId == 76 and timeIdx == 3) or (edgeId == 385 and timeIdx == 3) or (edgeId == 260 and timeIdx == 2) or (edgeId == 0 and timeIdx == 0):
					# print('s:',s,',t:',t, ',timeIdx:',timeIdx)
					# print('speedForTEC=',speedForTEC,',avgSpeed=',avgSpeed, ',speedLimit=',speedLimit, ',vehicleCount=',vehicleCount,',lengthInFoot=',lengthInFoot,',vehicleCount=',vehicleCount, ',TEC=',TEC)
				# emissionMatrix[edgeId][timeIdx] = TEC 
				emissionMatrix[edgeId][timeIdx] = TECPerHour
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
# print('index of 4349 is ',nodes.index('4349'))
# print('distMatrix of 4349\n',distMatrix[nodes.index('4349'),:])
numNodes = G.number_of_nodes()
#single objective shortest path:
startNode = '4320'
endNode = '4551'
startIdx = nodes.index(startNode)
endIdx = nodes.index(endNode)
problemSize = numNodes * numNodes
astarPath = nx.astar_path(G, startNode,endNode, weight='length')
print('A* path:\n',astarPath)
# minTECPathExample = ['4320', '4364', '4349', '4350', '4362', '4564', '4581', '4582', '3181', '3978', '3186', '3187', '3189', '3190', '4375', '3989', '3979', '3990', '4373', '3987', '2651', '3986', '4559', '4562', '4561', '4551']
# print('number of nodes:',G.number_of_nodes())
# print('number of edges:',G.number_of_edges())
# astartEdges = [(astarPath[n],astarPath[n+1]) for n in range(len(astarPath)-1)]
# minTECEdges = [(minTECPathExample[n],minTECPathExample[n+1]) for n in range(len(minTECPathExample)-1)]
# currTimeIdx = 0
# total_travelTime = 0
# for i in range(len(astarPath)-1):
# 	currNode = astarPath[i]
# 	nextNode = astarPath[i+1]
# 	edgeId = edges.index((currNode, nextNode))
# 	print(currNode, nextNode, ' ID:', edgeId, ',currTimeIdx:',currTimeIdx, ',length:',G.edges[currNode, nextNode]['length'], ',speedLimit:',G.edges[currNode, nextNode]['speedLimit'])
# 	print('travelTimeMatrix',travelTimeMatrix[edgeId][currTimeIdx])
# 	print('emissionMatrix', emissionMatrix[edgeId][currTimeIdx])
# 	total_travelTime = total_travelTime + travelTimeMatrix[edgeId][currTimeIdx]
# 	currTimeIdx = math.floor(total_travelTime/300) #every time instance is 5 minutes (300 seconds)
# print('minTECPathExample:\n',minTECPathExample)
# currTimeIdx = 0
# total_travelTime = 0
# for i in range(len(minTECPathExample)-1):
# 	currNode = minTECPathExample[i]
# 	nextNode = minTECPathExample[i+1]
# 	edgeId = edges.index((currNode, nextNode))
# 	print(currNode, nextNode, ' ID:', edgeId, ',currTimeIdx:',currTimeIdx, ',length:',G.edges[currNode, nextNode]['length'], ',speedLimit:',G.edges[currNode, nextNode]['speedLimit'])
# 	print('travelTimeMatrix',travelTimeMatrix[edgeId][currTimeIdx])
# 	print('emissionMatrix', emissionMatrix[edgeId][currTimeIdx])
# 	total_travelTime = total_travelTime + travelTimeMatrix[edgeId][currTimeIdx]
# 	currTimeIdx = math.floor(total_travelTime/300) #every time instance is 5 minutes (300 seconds)

for iteration in range(10):
	print('Iteration ',iteration)
	for exp in range(1,2):
		if exp == 0:
			dynamicFlag = DYNAMIC_NO
			print('Static Path Finding...')
		else:
			dynamicFlag = DYNAMIC_YES
			print('Dynamic Path Finding...')
		astrarScores = fitness_scores(astarPath, dynamicFlag) 
		print('A* path scores:\n',astrarScores)
		#NSGAII 
		initPopulationSize = 100
		numGenerations = 50
		tournamentSize = 2 
		numObjectives = 4 
		numConfigs = 4
		population = create_population()
		# print('population:\n',population)
		for config in range(3,numConfigs):  #only take the last config now
			# print('NSGAII Configuration ',config)
			for generation in range(numGenerations):
				if generation % 10 == 0:
					print('Generation %i\n'%generation)
				if config == 0:
					population = breed_population(population, dynamicFlag) #default: NBX with no mutation 
				if config == 1: 
					population = breed_population(population, dynamicFlag, CROSSOVER_MNX, MUTATION_NO) #MNX with no mutation
				if config == 2:
					population = breed_population(population, dynamicFlag, CROSSOVER_MNX, MUTATION_YES, 0.5) #MNX with mutation prob 0.5
				if config == 3:
					population = breed_population(population, dynamicFlag, CROSSOVER_MNX, MUTATION_YES, 0.8) #MNX with mutation prob 0.8		
				scores = score_population(population, dynamicFlag)
				population = build_pareto_population(population, scores)
			#get final pareto front
			scores = score_population(population, dynamicFlag)
			populationIds = np.arange(len(population))
			# print('before final pareto front: len of population:', len(population))
			paretoFront = identify_pareto(scores, populationIds)
			paretoFrontList = list(paretoFront)
			finalSolutions = []
			finalScores = []
			for i in range(len(population)):
				if i in paretoFrontList:
					finalSolutions.append(population[i])
					finalScores.append(scores[i])
			# print('after final pareto front: len of population:', len(finalSolutions))
			# print('final solutions:')
			# for i in range(len(finalSolutions)):
			# 	print('path ',i,':',finalSolutions[i])
			# 	print('scores:',finalScores[i])

			finalScores = np.asarray(finalScores)
			x = np.around(finalScores[:,0], decimals = 3)
			y = np.round(finalScores[:,1], decimals = 3)
			z = finalScores[:,2]
			c = np.round(finalScores[:,3], decimals = 3)
			finalScores2 = np.array(list(zip(x,y,z,c)))
			title = 'From Node 4320(Hinnerup) to Node 4551(Hasselager) in Aarhus, Denmark\n'
			if config == 0:
				title = title + 'NSGAII - Single Point Crossover, No Mutation'
			if config == 1:
				title = title + 'NSGAII - Multi Point Crossover, No Mutation'
			if config == 2:
				title = title + 'NSGAII - Multi Point Crossover, Mutation Prob 0.5'		
			if config == 3:
				title = title + 'NSGAII - Multi Point Crossover, Mutation Prob 0.8'			

			#visualization 1: 3D + color plot 
			# fig = plt.figure()
			# # fig.set_size_inches(20, 20)
			# ax = fig.add_subplot(111, projection='3d')				
			# fig.suptitle(title)
			# ax.set_xlabel('Objective 1 - min distance (KM)')
			# ax.set_ylabel('Objective 2 - min turns')
			# ax.set_zlabel('Objective 3 - min travel time (minutes)')
			# # ax.set_xlim([20, 40])  #hard code these limits here for now for fair visual comparison - not flexible
			# # ax.set_ylim([8, 32])
			# # ax.set_zlim([15, 70])

			# # img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
			# pnt3d = ax.scatter(x, y, z, c=c)
			# cbar = plt.colorbar(pnt3d)
			# cbar.set_label("Objective 4 - min total vehicle emission cost (TEC)")

			# # fig.colorbar(img)
			# figName = 'MultObjectivePathFindingConfig'+str(config)+'.png'
			# plt.savefig(figName)
			# plt.show()	

			#visualization 2: parallel cooridinates
			# fig2 = go.Figure(data=
			# 	go.Parcoords(
			# 		line=dict(
			# 			color=c, 
			# 			# colorscale = 'Electric', 
			# 			showscale = True),
			# 		dimensions = list([
			# 			dict(label= 'Distance (KM)', values = x),
			# 			dict(label='Turns', values = y),
			# 			dict(label='Time (Minutes)', values = z)
			# 			])
			# 		)
			# 	)	
			# fig2.show()
			# mooDfObj = pd.DataFrame(finalScores, columns = ['Distance','Turns','Time','TEC'])
			mooDfObj = pd.DataFrame(finalScores2, columns = ['TEC', 'Time','Turns','Distance'])
			fig3 = px.parallel_coordinates(
				mooDfObj, 
				color="TEC", 
				labels={"TEC":"Total Emission Cost (TEC)", "Time":"Time (Minutes)", "Turns":"# Turns", "Distance":"Distance (KM)"}
				)
			fig3.show()
			# figName = "images/fig3_Iter"+str(iteration)+".png"
			# fig3.write_image(figName)
		G2 = nx.DiGraph()
		#path with minimum emission 
		# minTECIndices = np.where(finalScores[:,3] == np.amin(finalScores[:,3]))
		minTECIndices = np.where(finalScores[:,0] == np.amin(finalScores[:,0]))
		minTECIndex = minTECIndices[0][0]
		minTimeIndices = np.where(finalScores[:,1] == np.amin(finalScores[:,1]))
		minTimeIndex = minTimeIndices[0][0]
		minTurnIndices = np.where(finalScores[:,2] == np.amin(finalScores[:,2]))
		minTurnIndex = minTurnIndices[0][0]
		minDistanceIndices = np.where(finalScores[:,3] == np.amin(finalScores[:,3]))
		minDistanceIndex = minDistanceIndices[0][0]	
		# print('minTECIndices:',minTECIndices, ',minTECIndices[0]:',minTECIndices[0],',minTECIndex:',minTECIndex)
		minTECPath = finalSolutions[minTECIndex]
		minTimePath = finalSolutions[minTimeIndex]
		minTurnPath = finalSolutions[minTurnIndex]
		minDistancePath = finalSolutions[minDistanceIndex]
		routes=[]
		edges=[]
		routes.append(astarPath)
		routes.append(minTECPath)
		routes.append(minTimePath)
		routes.append(minTurnPath)
		routes.append(minDistancePath)
		for r in routes:
			route_edges = [(r[n],r[n+1]) for n in range(len(r)-1)]
			G2.add_nodes_from(r)
			G2.add_edges_from(route_edges)
			edges.append(route_edges)
		# print('edges[0]:',edges[0])
		# print('edges[1]:',edges[1])
		pos={}
		for node in G2.nodes():
			#get position data from the main graph
			pos[node] = [G.node[node]['pos'][0],G.node[node]['pos'][1]]	
		# pos = nx.spring_layout(G2)
		lab=dict(zip(G2,G2.nodes())) #use node id as label
		nx.draw(G2,pos=pos,node_size=30, with_labels=False) #first suppress the default labels
		nx.draw_networkx_nodes(G2, pos=pos, nodelist=minDistancePath,node_color='blue', node_size=30)
		nx.draw_networkx_nodes(G2, pos=pos, nodelist=minTurnPath,node_color='orange', node_size=30)
		nx.draw_networkx_nodes(G2, pos=pos, nodelist=minTimePath,node_color='yellow', node_size=30)
		nx.draw_networkx_nodes(G2, pos=pos, nodelist=minTECPath,node_color='green', node_size=30)
		nx.draw_networkx_nodes(G2, pos=pos, nodelist=astarPath,node_color='red', node_size=30)
		nx.draw_networkx_nodes(G2, pos=pos, nodelist=[startNode, endNode],node_color='red', node_size=50)
		# nx.draw_networkx_labels(G2, pos=pos)
		nx.draw_networkx_labels(G2,pos, lab, font_size=8) #then draw with the custom labels
		# nx.draw_networkx_edges(G2, pos=pos)
		# nx.draw_networkx_edges(G2, pos=pos, edgeList = edges[1], edge_color = 'b')
		# nx.draw_networkx_edges(G2, pos=pos, edgeList = edges[0], edge_color = 'r')
		plt.savefig("images/DynamicPaths"+str(iteration)+".png")
		# plt.show()
		print('minTECPath:',minTECPath)
		print('minTECPath scores', finalScores[minTECIndex,:])
		print('minTimePath:',minTimePath)
		print('minTimePath scores', finalScores[minTimeIndex,:])
		print('minTurnPath:',minTurnPath)
		print('minTurnPath scores', finalScores[minTurnIndex,:])
		print('minDistancePath:',minDistancePath)
		print('minDistancePath scores', finalScores[minDistanceIndex,:])

print('done')




