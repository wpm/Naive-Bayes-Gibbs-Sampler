#!/usr/bin/python

import numpy
from numpy import array
from numpy import empty
from numpy import ones
from numpy import zeros
from numpy.random import dirichlet
from numpy.random import sample


def discrete_random_sample(distribution):
	uniform = sample()
	for i, p in enumerate(distribution):
		if uniform < p:
			return i
		uniform -= p
	return i


def generate_corpus(c, v, r, n):
	pi = dirichlet([1]*c, 1)[0]
	thetas = dirichlet([1]*v, c)
	corpus = empty((n,v), numpy.int)
	labels = empty(n, numpy.int)
	for i in xrange(n):
		c = discrete_random_sample(pi)
		labels[i] = c
		theta = thetas[c]
		w = zeros(v, numpy.int)
		for j in xrange(r):
			k = discrete_random_sample(theta)
			w[k] += 1
		corpus[i] = w
	return pi, thetas, corpus, labels


def initialize_estimation(hyp_pi, hyp_thetas, n):
	pi = dirichlet(hyp_pi, 1)[0]
	c, v = hyp_thetas.shape
	thetas = empty(hyp_thetas.shape)	
	for i in xrange(c):		
		thetas[i] = dirichlet(hyp_thetas[i], 1)[0]	
	labels = array([discrete_random_sample(pi) for i in xrange(n)])
	return pi, thetas, labels
	
	
def iterate_estimation(hyp_pi, hyp_thetas, pi, thetas, corpus, labels):
	c = len(pi)		# number of categories
	v = len(thetas)	# vocabulary size
	# Get counts for the classes.
	counts = [[0]*v]*c
	for i, document in enumerate(corpus):
		label = labels[i]
		


if __name__ == "__main__":
	# Generate data set
	c = 2	# number of categories
	v = 4	# vocabulary size
	r = 100	# document length
	n = 5	# dataset size
	
	true_pi, true_theta, corpus, true_labels = generate_corpus(c, v, r, n)
	print "true pi %s" % true_pi
	print "true thetas\n%s" % true_theta
	print "true labels %s" % true_labels
	print "corpus\n%s" % corpus

	print "\n\n"
	pi, thetas, labels = initialize_estimation(ones(c, numpy.int), ones((c,v), numpy.int), n)
	print "pi %s" % pi
	print "thetas\n%s" % thetas
	print "labels %s" % labels
	