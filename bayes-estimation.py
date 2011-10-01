#!/usr/bin/python

from numpy.random import beta
from numpy.random import dirichlet
from numpy.random import sample

def sample_from_distribution(distribution):
	uniform = sample()
	for i, p in enumerate(distribution):
		if uniform < p:
			return i
		uniform -= p
	return i


def generate_corpus(c, v, r, n):
	pi = dirichlet([1]*c, 1)[0]
	thetas = dirichlet([1]*v, c)
	corpus = []
	labels = []
	for i in xrange(n):
		c = sample_from_distribution(pi)
		labels.append(c)
		theta = thetas[c]
		w = [0]*v
		for j in xrange(r):
			k = sample_from_distribution(theta)
			w[k] += 1
		corpus.append(w)
	return pi, thetas, corpus, labels


def initialize_estimation(hyp_pi, hyp_thetas, n):
	pi = dirichlet(hyp_pi, 1)[0]
	thetas = [dirichlet(hyp_theta, 1)[0] for hyp_theta in hyp_thetas]
	labels = [sample_from_distribution(pi) for i in xrange(n)]
	return pi, thetas, labels


if __name__ == "__main__":
	# Generate data set
	c = 2	# number of categories
	v = 4	# vocabulary size
	r = 100	# document length
	n = 4	# dataset size
	
	true_pi, true_theta, corpus, true_labels = generate_corpus(c, v, r, n)
	print "true pi %s" % true_pi
	print "true theta\n%s" % true_theta
	print "true labels %s" % true_labels
	print "corpus\n%s" % corpus

	print "\n\n"
	pi, thetas, labels = initialize_estimation([1]*c, [[1]*v]*c, n)
	print "pi %s" % pi
	print "thetas\n%s" % thetas
	print "labels %s" % labels
	