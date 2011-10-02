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


class GibbsSampler(object):
	def __init__(self, hyp_pi, hyp_thetas, corpus):
		"""
		Initialize the Gibbs sampler

		@param hyp_pi: category hyperparameter
		@type hyp_pi: array of float
		@param hyp_thetas: word distribution in category hyperparameter
		@type hyp_thetas: matrix of float
		@param corpus: word frequencies per document
		@type corpus: matrix of integers
		"""
		self.hyp_pi = hyp_pi
		self.hyp_thetas = hyp_thetas
		self.corpus = corpus		

	def __str__(self):
		return "hyper pi\n%s\nhyper thetas\n%s\nthetas\n%slabels %s" % \
			(self.hyp_pi, self.hyp_thetas, self.thetas, self.labels)
	
	def run(self, iterations = 10, burn_in = 0, lag = 0):
		"""
		Run the Gibbs sampler
		
		@param iterations: number of iterations to run
		@type iterations: integer
		@param burn_in: number of burn in iterations to ignore before returning results
		@type burn_in: integer
		@param lag: number of iterations to skip between returning values
		@type lag: integer
		"""
		self._initialize_gibbs_sampler()
		lag_counter = lag
		for iteration in xrange(iterations):
			self._iterate_gibbs_sampler()
			if burn_in > 0:
				burn_in -= 1
			else:
				if lag_counter > 0:
					lag_counter -= 1
				else:
					lag_counter = lag
					yield iteration, self.thetas, self.labels
	
	def _initialize_gibbs_sampler(self):
		"""
		Initialize the Gibbs sampler
		
		This sets the initial values of the C{labels} and C{thetas} parameters.		
		"""
		pi = dirichlet(self.hyp_pi, 1)[0]
		c, v = hyp_thetas.shape
		self.thetas = empty(self.hyp_thetas.shape)
		for i in xrange(c):
			self.thetas[i] = dirichlet(self.hyp_thetas[i], 1)[0]
		self.labels = array([multinomial_sample(pi) \
			for i in xrange(self.corpus.shape[0])])
	
	def _iterate_gibbs_sampler(self):
		"""
		Perform a Gibbs sampling iteration.
		
		This updates the values of the C{labels} and C{thetas} parameters.		
		"""
		documents = self.corpus.shape[0]	# corpus size
		vocaublary = self.corpus.shape[1]	# vocabulary size
		categories = self.hyp_pi.size		# number of categories
		# Get class counts and word counts for the classes.
		category_counts = empty(categories, int)
		word_counts = empty((categories, vocaublary), int)
		for category in xrange(categories):
			category_counts[category] = count_nonzero(self.labels == category)
			word_counts[category] = self.corpus[nonzero(self.labels == category)].transpose().sum(1)

		# Estimate the new document labels.
		for document in xrange(documents):
			category = self.labels[document]
			word_counts[category] -= corpus[document]
			category_counts[category] -= 1
			posterior_pi = empty(categories)
			# Calculate label posterior for a single document.
			for category in xrange(categories):
				label_factor = \
					(word_counts[category].sum() + self.hyp_pi[category] - 1.0)/ \
					(word_counts.sum() + self.hyp_pi.sum() - 1.0)	# Writing 1.0 forces a cast
																	# to float.
				word_factor = (self.thetas[category]**word_counts[category]).prod()
				posterior_pi[category] = label_factor * word_factor
			# Select a new label for the document.
			posterior_pi /= posterior_pi.sum()
			new_category = multinomial_sample(posterior_pi)
			self.labels[document] = new_category
			word_counts[new_category] += self.corpus[document]
			category_counts[new_category] += 1

		# Estimate the new word count distributions.
		t = word_counts + self.hyp_thetas
		for theta_index in xrange(categories):
			self.thetas[theta_index] = dirichlet(t[theta_index], 1)[0]


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
	
	# Create the Gibbs sampler.
	hyp_pi = ones(c, int)			# uninformed label prior
	hyp_thetas = ones((c,v), int)	# uninformed word prior
	sampler = GibbsSampler(hyp_pi, hyp_thetas, corpus)
	
	# Run the Gibbs sampler.
	for iteration, thetas, labels in sampler.run(3):
		print "\nIteration %d" % (iteration+1)
		print "thetas\n%s" % thetas
		print "labels %s" % labels
