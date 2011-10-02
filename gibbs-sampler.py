#!/usr/bin/env python

from numpy import array, count_nonzero, empty, ones, nonzero, zeros
from numpy.random import dirichlet, multinomial


def multinomial_sample(distribution):
	"""
	Sample a random integer according to a multinomial distribution.
	
	@param distribution: probabilitiy distribution
	@type distribution: array
	@return: integer in the range 0 to the length of distribution
	@rtype: integer
	"""
	return nonzero(multinomial(1, distribution))[0][0]

def generate_corpus(c, v, r, n, hyp_pi = None, hyp_thetas = None):
	"""
	Create model parameters and sample data for a corpus of labeled documents.
	
	@param c: number of categories
	@type c: integer
	@param v: vocabulary size
	@type v: integer
	@param r: document length
	@type r: integer
	@param: n dataset size
	@type n: integer
	@param hyp_pi: optional category hyperparamter, default uninformative
	@type hyp_pi: list or None
	@param hyp_thetas: optional word count hyperparamter, default uninformative
	@type hyp_thetas: list or None
	@return: category distribution, word distributions per category, documents,
			document labels
	@rtype: tuple
	"""
	if hyp_pi == None:
		hyp_pi = [1]*c
	if len(hyp_pi) != c:
		raise Exception()
	if hyp_thetas == None:
		hyp_thetas = [1]*v
	if len(hyp_thetas) != v:
		raise Exception()
	pi = dirichlet(hyp_pi, 1)[0]
	thetas = dirichlet(hyp_thetas, c)
	corpus = empty((n,v), int)
	labels = empty(n, int)
	for i in xrange(n):
		c = multinomial_sample(pi)
		labels[i] = c
		theta = thetas[c]
		w = zeros(v, int)
		for _ in xrange(r):
			k = multinomial_sample(theta)
			w[k] += 1
		corpus[i] = w
	return pi, thetas, corpus, labels


def initialize_gibbs_sampling(hyp_pi, hyp_thetas, corpus):
	"""
	Initialize a Gibbs sampling run
	
	@param hyp_pi: category hyperparameter
	@type hyp_pi: array of float
	@param hyp_thetas: word distribution in category hyperparameter
	@type hyp_thetas: matrix of float
	@param corpus: word frequencies per document
	@type corpus: matrix of integers
	@return: initial category distribution, word distributions, and labels
	@rtype: tuple
	"""
	pi = dirichlet(hyp_pi, 1)[0]
	c, v = hyp_thetas.shape
	thetas = empty(hyp_thetas.shape)
	for i in xrange(c):
		thetas[i] = dirichlet(hyp_thetas[i], 1)[0]
	labels = array([multinomial_sample(pi) for i in xrange(corpus.shape[0])])
	return pi, thetas, labels


def iterate_gibbs_sampling(hyp_pi, hyp_thetas, thetas, labels, corpus):
	"""
	Perform a Gibbs sampling iteration
	
	@param hyp_pi: category hyperparameter
	@type hyp_pi: array of float
	@param hyp_thetas: word distribution in category hyperparameter
	@type hyp_thetas: matrix of float
	@param thetas: current word distribution in category parameter
	@type thetas: matrix of float
	@param labels: labels of the documents
	@type labels: array of int
	@param corpus: word frequencies per document
	@type corpus: matrix of integers
	@return: new category distribution, word distributions, and labels
	@rtype: tuple
	"""
	documents = corpus.shape[0]		# corpus size
	vocaublary = corpus.shape[1]	# vocabulary size
	categories = hyp_pi.size		# number of categories
	# Get class counts and word counts for the classes.
	category_counts = empty(categories, int)
	word_counts = empty((categories, vocaublary), int)
	for category in xrange(categories):
		category_counts[category] = count_nonzero(labels == category)
		word_counts[category] = corpus[nonzero(labels == category)].transpose().sum(1)
	
	# Estimate the new document labels.
	for document in xrange(documents):
		category = labels[document]
		word_counts[category] -= corpus[document]
		category_counts[category] -= 1
		posterior_pi = empty(categories)
		# Calculate label posterior for a single document.
		for category in xrange(categories):
			label_factor = \
				(word_counts[category].sum() + hyp_pi[category] - 1.0)/ \
				(word_counts.sum() + hyp_pi.sum() - 1.0)	# Writing 1.0 forces a cast
															# to float.
			word_factor = (thetas[category]**word_counts[category]).prod()
			posterior_pi[category] = label_factor * word_factor
		# Select a new label for the document.
		posterior_pi /= posterior_pi.sum()
		new_category = multinomial_sample(posterior_pi)
		labels[document] = new_category
		word_counts[new_category] += corpus[document]
		category_counts[new_category] += 1
	
	# Estimate the new word count distributions.
	t = word_counts + hyp_thetas
	for theta_index in xrange(categories):
		thetas[theta_index] = dirichlet(t[theta_index], 1)[0]
	
	return thetas, labels


if __name__ == "__main__":
	# Generate data set
	c = 3	# number of categories
	v = 4	# vocabulary size
	r = 100	# document length
	n = 10	# dataset size
	
	true_pi, true_theta, corpus, true_labels = generate_corpus(c, v, r, n)
	print "true pi %s" % true_pi
	print "true thetas\n%s" % true_theta
	print "true labels %s" % true_labels
	print "corpus\n%s" % corpus
	
	hyp_pi = ones(c, int)			# uninformed label prior
	hyp_thetas = ones((c,v), int)	# uninformed word prior
	pi, thetas, labels = initialize_gibbs_sampling(hyp_pi, hyp_thetas, corpus)
	print "\nInitialize"
	print "pi %s" % pi
	print "thetas\n%s" % thetas
	print "labels %s" % labels

	for i in xrange(3):
		thetas, labels = iterate_gibbs_sampling(
			hyp_pi, hyp_thetas,
			thetas, labels,
			corpus)
		print "\nIteration %d" % (i+1)
		print "thetas\n%s" % thetas
		print "labels %s" % labels
