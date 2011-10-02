#!/usr/bin/env 	python

from numpy import array
from numpy import count_nonzero
from numpy import empty
from numpy import ones
from numpy import nonzero
from numpy import zeros
from numpy.random import dirichlet
from numpy.random import sample


def discrete_random_sample(distribution):
	"""
	Return a random integer according to a discrete distribution.
	
	@param distribution: probabilitiy distribution
	@type distribution: array
	@return: integer in the range 0 to the length of distribution
	@rtype: integer
	"""
	uniform = sample()
	for i, p in enumerate(distribution):
		if uniform < p:
			return i
		uniform -= p
	return i


def generate_corpus(c, v, r, n):
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
	@return: category distribution, word distributions per category, documents,
			document labels
	@rtype: tuple
	"""
	pi = dirichlet([1]*c, 1)[0]
	thetas = dirichlet([1]*v, c)
	corpus = empty((n,v), int)
	labels = empty(n, int)
	for i in xrange(n):
		c = discrete_random_sample(pi)
		labels[i] = c
		theta = thetas[c]
		w = zeros(v, int)
		for _ in xrange(r):
			k = discrete_random_sample(theta)
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
	labels = array([discrete_random_sample(pi) for i in xrange(corpus.shape[0])])
	return pi, thetas, labels


def iterate_gibbs_sampling(hyp_pi, hyp_thetas, pi, thetas, corpus, labels):
	"""
	Perform a Gibbs sampling iteration
	
	@param hyp_pi: category hyperparameter
	@type hyp_pi: array of float
	@param hyp_thetas: word distribution in category hyperparameter
	@type hyp_thetas: matrix of float
	@param pi: current category parameter
	@type pi: array of float
	@param thetas: current word distribution in category parameter
	@type thetas: matrix of float
	@param corpus: word frequencies per document
	@type corpus: matrix of integers
	@return: new category distribution, word distributions, and labels
	@rtype: tuple
	"""
	v = corpus.shape[1]	# vocabulary size
	c = pi.size			# number of categories
	# Get class counts and word counts for the classes.
	category_counts = empty(c)
	word_counts = empty((c, v), int)
	for label in xrange(c):
		category_counts[label] = count_nonzero(labels == label)
		word_counts[label] = corpus[nonzero(labels == label)].transpose().sum(1)
	# Get the posterior probabilites of the documents.
	document_posteriors = (thetas**word_counts).prod(1)
	return pi, thetas, labels


if __name__ == "__main__":
	# Generate data set
	c = 2	# number of categories
	v = 4	# vocabulary size
	r = 100	# document length
	n = 3	# dataset size
	
	true_pi, true_theta, corpus, true_labels = generate_corpus(c, v, r, n)
	print "true pi %s" % true_pi
	print "true thetas\n%s" % true_theta
	print "true labels %s" % true_labels
	print "corpus\n%s" % corpus
	
	print "\n\n"
	hyp_pi = ones(c, int)			# uninformed label prior
	hyp_thetas = ones((c,v), int)	# uninformed word prior
	pi, thetas, labels = initialize_gibbs_sampling(hyp_pi, hyp_thetas, corpus)
	print "pi %s" % pi
	print "thetas\n%s" % thetas
	print "labels %s" % labels
	
	pi, thetas, labels = iterate_gibbs_sampling(
		hyp_pi, hyp_thetas,
		pi, thetas,
		corpus, labels)
	