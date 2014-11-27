'''
HDP implementation by Joon Hee Kim (joonheekim@gmail.com)

alpha1	 	parameter for creating new table
beta1 		parameter for topic smoothing
phi 		topic
phi_sum 	sum of phi
gamma 		parameter for creating new topic
no_below	same as in gensim library
no_above	same as in gensim library

data format for corpus should consists of a file where each line is a document, and each token is separated by a tab
an example is in /corpus/nips_5_0.2/nips.txt
where 5 = no_below and 0.2 = no_above
'''

import re, gensim, pickle, logging, random, time, math, os, bisect
import numpy as np
from copy import copy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_dir = '../data/'
corpus_dir = '../data/corpus/'
utils_dir = '../data/utils/'
exp_dir = '../experiment/'

class Table:
	def __init__(self):
		self.words = []
		self.topic = None

class Topic:
	def __init__(self, num_word, beta, index):
		# self.index = index
		self.table_count = 0
		self.phi = np.ones((num_word)) * beta
		self.phi_sum = beta * num_word
		self.index = index

class Document:
	def __init__(self, docID, time, words):
		self.docID = docID
		self.time = time
		self.words = words
		self.word2table = [None] * len(words)
		self.tables = []

class Model:
	def __init__(self, alpha1, beta1, alpha2, beta2, gamma, data_type, iteration, no_below, no_above, max_topic):
		self.alpha1 = alpha1
		self.beta1 = beta1
		self.alpha2 = alpha2
		self.beta2 = beta2
		self.gamma = gamma
		self.max_topic = max_topic
		self.data_type = data_type
		self.iteration = iteration
		self.no_below = no_below
		self.no_above = no_above
		self.param = data_type + '_' + str(alpha1) + '_' + str(beta1) + '_'+ str(alpha2) + '_' + str(beta2) + '_' + str(gamma) + '_' + str(max_topic)

	def load_data(self):
		self.dictionary = gensim.corpora.dictionary.Dictionary.load(utils_dir + self.data_type + '_' + str(self.no_below) + '_' + str(self.no_above) + '.dic')
		self.num_word = len(self.dictionary)
		self.corpus = []
		length = 0
		num_doc = 0

		logging.info('reading data')	
		f = open(corpus_dir + self.data_type + '_' + str(self.no_below) + '_' + str(self.no_above) + '/' + self.data_type + '.txt', 'r')
		for line in f:
			words = [int(word) for word in line.strip().split()]
			doc = Document(None, None, words)
			self.corpus.append(doc)
			num_doc += 1
			length += len(words)
		f.close()
		logging.info('average document length:' + str(length / float(num_doc)))

		self.available_index = range(self.max_topic)
		self.index2topic = {} 											# {topic_index: topic}
		self.topic_count = np.zeros((self.max_topic))					# topic count (df)

	def run(self):
		for i in range(self.iteration):
			logging.info('iteration: ' + str(i) + '\t processing: ' + str(len(self.corpus)) + ' documents')
			for document in self.corpus:
				self.process_document(document)
			self.print_count()
			self.print_state(i)
		
	def top_words(self, vector, n):
		vector = copy(vector)
		result = ''
		for i in range(n):
			argmax = np.argmax(vector)
			value = vector[argmax]
			vector[argmax] = -1
			result += self.dictionary[argmax]
			result += '\t'
			result += ("%.3f"%value)
			result += '\t'
		return result

	def choose(self, roulette):
		total = sum(roulette)
		arrow = total * random.random()
		roulette = np.cumsum(roulette)
		return bisect.bisect(np.cumsum(roulette), arrow)

	def print_topics(self):
		string = ''
		for index, topic in self.index2topic.items():
			string += (str(index) + '\t' + str(topic.table_count) + '\t' + (self.top_words(topic.phi, 10) + '\n'))
		return string

	def print_state(self, i):
		logging.info('printing state\t' + self.param)
		if not os.path.isdir(exp_dir + self.param):
			os.mkdir(exp_dir + self.param)
		write_file = open(exp_dir + self.param + '/' + str(i) + '.txt', 'w')
		write_file.write(self.print_topics())
		write_file.close()

	def print_count(self):
		num_topics = len(self.index2topic)
		num_tables = 0
		for index, topic in self.index2topic.items():
			num_tables += topic.table_count
		logging.info('num_topics: ' + str(num_topics))
		logging.info('num_tables: ' + str(num_tables))
		logging.info('num_average_tables: ' + str(num_tables / float(len(self.corpus))))

	def process_document(self, document):
		if len(document.tables) == 0:
			random.shuffle(document.words)

		# table assignment
		for i in range(len(document.words)):
			word = document.words[i]

			# de-assignment
			old_table = document.word2table[i]

			# if first assignment, pass de-assignment
			if old_table == None:
				pass
			else:
				# remove previous assignment related to word
				old_table.words.remove(word)
				old_topic = old_table.topic
				old_topic.phi[word] -= 1
				old_topic.phi_sum -= 1
				
				# if old_table has no word, remove it
				if len(old_table.words) == 0:
					document.tables.remove(old_table)
					old_topic.table_count -= 1

					for other_table in document.tables:
						if other_table != old_table:
							other_topic = other_table.topic
							self.topic_count[other_topic.index] -= 1
							self.topic_count[old_topic.index] -= 1

					# remove topic if necessary
					if old_topic.table_count == 0:
						self.remove_topic(old_topic)

			# table assignment
			roulette = np.zeros((len(document.tables) + 1))
			for j in range(len(document.tables)):
				table = document.tables[j]
				roulette[j] = (table.topic.phi[word] / table.topic.phi_sum) * len(table.words)
			roulette[-1] = self.alpha1 / self.num_word
			new_table_index = self.choose(roulette)

			# error case
			if new_table_index == -1:
				print 'error 1'
				exit(-1)

			# create new table if last index is chosen
			if new_table_index == len(document.tables):
				new_table = Table()
				new_topic = self.get_topic_for_table(new_table, document)
				for other_table in document.tables:
					other_topic = other_table.topic
					self.topic_count[other_topic.index] += 1
					self.topic_count[new_topic.index] += 1
				document.tables.append(new_table)				
				new_table.topic = new_topic
				new_topic.table_count += 1
			else:
				new_table = document.tables[new_table_index]
				new_topic = new_table.topic
			new_table.words.append(word)
			new_topic.phi[word] += 1
			new_topic.phi_sum += 1
			document.word2table[i] = new_table

		# self.index2topic = {} 											# {topic_index: topic}
		# self.topic_count = np.zeros((self.max_topic))					# topic count (df)

		# topic assignment
		for i in range(len(document.tables)):
			# de-assignment
			table = document.tables[i]
			old_topic = table.topic
			for other_table in document.tables:
				if other_table != table:
					other_topic = other_table.topic
					self.topic_count[other_topic.index] += 1
					self.topic_count[old_topic.index] += 1

			for word in table.words:
				old_topic.phi[word] -= 1
			old_topic.phi_sum -= len(table.words)
			old_topic.table_count -= 1	

			if old_topic.table_count == 0:
				self.remove_topic(old_topic)

			new_topic = self.get_topic_for_table(table, document)
			table.topic = new_topic
			
			for word in table.words:
				new_topic.phi[word] += 1
			new_topic.phi_sum += len(table.words)
			new_topic.table_count += 1
			for other_table in document.tables:
				if other_table != table:
					other_topic = other_table.topic
					self.topic_count[other_topic.index] += 1
					self.topic_count[new_topic.index] += 1


	
	def remove_topic(self, topic):
		index = topic.index
		del self.index2topic[index]
		self.available_index.append(index)

	def get_topic_for_table(self, table, document):	
		# in some extreme case, if there are too many words in one table
		# we can have float overflow error, so for now we just limit the words in one table
		# that we use in sampling by this number. normally we don't have case where
		# words exceed 75 in one table, so it doesn't matter.
		# but still we should fix this later.
		word_limit = 75

		# self.index2topic = {} 											# {topic_index: topic}
		# self.topic_count = np.zeros((self.max_topic))					# topic count (df)

		roulette = np.zeros((len(self.index2topic) + 1))
		i = 0
		for index, topic in self.index2topic.items():
			roulette[i] = topic.table_count
			for word in table.words[:word_limit]:
				roulette[i] *= (topic.phi[word] / topic.phi_sum * self.num_word)
			i += 1
		if len(self.available_index) == 0:
			roulette[i] = 0
		else:
			roulette[i] = self.gamma
		# print roulette
		topic_index = self.choose(roulette)
		if topic_index == -1:
			logging.info('error in get_topic_for_table')
			logging.info('len(table.words):' + str(len(table.words)))
			exit(-1)
		if topic_index == len(roulette) - 1:
			# create new topic
			new_index = self.available_index.pop(0)
			topic = Topic(self.num_word, self.beta1, new_index)
			self.index2topic[new_index] = topic
		else:
			topic = self.index2topic[self.index2topic.keys()[topic_index]]
		return topic

def test_nips():
	alpha1 = 0.2
	beta1 = 0.1
	alpha2 = None
	beta2 = None
	gamma = 1
	max_topic = 1000
	data_type = 'nips'
	iteration = 1000
	no_below = 5
	no_above = 0.2
	model = Model(alpha1, beta1, alpha2, beta2, gamma, data_type, iteration, no_below, no_above, max_topic)

	model.load_data()
	model.run()

test_nips()