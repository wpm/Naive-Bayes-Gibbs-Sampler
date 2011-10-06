"""
Implementation of the Gibbs sampler for Naive Bayes document classification described
in U{Resnik and Hardisty 2010, "Gibbs Sampling for the Uninitiated
<http://drum.lib.umd.edu/handle/1903/10058>}.
"""

from numpy import array, empty, ones, nonzero, repeat, zeros
from numpy import count_nonzero, exp, inf, log, logaddexp
from numpy.random import dirichlet, multinomial


class GibbsSampler(object):
	"""Naive Bayes Gibbs sampler
	
	Given a set of hyperparamters and a corpus of document word counts, this objects
	runs Gibbs sampling to assign labels to the documents using a Naive Bayes model.
	"""
	def __init__(self, hyp_pi, hyp_thetas, corpus):
		"""Initialize the Gibbs sampler

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


	def estimate_labels(self, iterations = 10, burn_in = 0, lag = 0):
		"""Estimate the document labels.

		Run the Gibbs sampler and use the expected value of the labels as the label
		estimates.

		@param iterations: number of iterations to run
		@type iterations: integer
		@param burn_in: number of burn in iterations to ignore before returning results
		@type burn_in: integer
		@param lag: number of iterations to skip between returning values
		@type lag: integer
		@return: document label estimates
		@rtype: array
		"""
		estimated_labels = zeros(self._categories(), int)
		for iteration, thetas, labels in sampler.run(iterations, burn_in, lag):
			estimated_labels += labels
		return estimated_labels/iterations

	def run(self, iterations = 10, burn_in = 0, lag = 0):
		"""Run the Gibbs sampler

		@param iterations: number of iterations to run
		@type iterations: integer
		@param burn_in: number of burn in iterations to ignore before returning results
		@type burn_in: integer
		@param lag: number of iterations to skip between returning values
		@type lag: integer
		"""
		self._initialize_gibbs_sampler()
		lag_counter = lag
		iteration = 1
		while iteration <= iterations:
			self._iterate_gibbs_sampler()
			if burn_in > 0:
				burn_in -= 1
			else:
				if lag_counter > 0:
					lag_counter -= 1
				else:
					lag_counter = lag
					yield iteration, self.thetas, self.labels
					iteration += 1

	def _categories(self):
		"""
		@return: number of categories in the model
		@rtype: integer
		"""
		return self.hyp_pi.size

	def _documents(self):
		"""
		@return: number of documents in the corpus
		@rtype: integer
		"""
		return self.corpus.shape[0]

	def _vocabulary(self):
		"""
		@return: size of the vocabulary
		@rtype: integer
		"""
		return self.corpus.shape[1]

	def _initialize_gibbs_sampler(self):
		"""
		Initialize the Gibbs sampler

		This sets the initial values of the C{labels} and C{thetas} parameters.
		"""
		pi = log(dirichlet(self.hyp_pi, 1)[0])
		categories = self._categories()
		documents = self._documents()
		self.thetas = empty(self.hyp_thetas.shape)
		for category_index in xrange(categories):
			self.thetas[category_index] = \
				log(dirichlet(self.hyp_thetas[category_index], 1)[0])
		self.labels = array([multinomial_sample(pi) for _ in xrange(documents)])

	def _iterate_gibbs_sampler(self):
		"""Perform a Gibbs sampling iteration.

		This updates the values of the C{labels} and C{thetas} parameters.
		"""
		documents = self._documents()	# corpus size
		vocabulary = self._vocabulary()	# vocabulary size
		categories = self._categories()	# number of categories
		# Get class counts and word counts for the classes.
		category_counts = empty(categories, int)
		word_counts = empty((categories, vocabulary), int)
		for category_index in xrange(categories):
			category_counts[category_index] = \
				count_nonzero(self.labels == category_index)
			word_counts[category_index] = \
				self.corpus[nonzero(self.labels == category_index)].transpose().sum(1)

		# Estimate the new document labels.
		for document_index in xrange(documents):
			category_index = self.labels[document_index]
			word_counts[category_index] -= corpus[document_index]
			category_counts[category_index] -= 1
			posterior_pi = repeat(-inf, categories)
			# Calculate label posterior for a single document.
			for category_index in xrange(categories):
				num = word_counts[category_index].sum() + \
					self.hyp_pi[category_index] - 1.0
				if num != 0:
					den = word_counts.sum() + self.hyp_pi.sum() - 1.0
					label_factor = num/den
					word_factor = \
						(self.thetas[category_index]*word_counts[category_index]).sum()
					posterior_pi[category_index] = log(label_factor) + word_factor
			# Select a new label for the document.
			posterior_pi -= self._sum_log_array(posterior_pi)
			new_category = multinomial_sample(posterior_pi)
			self.labels[document_index] = new_category
			word_counts[new_category] += self.corpus[document_index]
			category_counts[new_category] += 1

		# Estimate the new word count distributions.
		t = word_counts + self.hyp_thetas
		for category_index in xrange(categories):
			self.thetas[category_index] = log(dirichlet(t[category_index], 1)[0])

	def _sum_log_array(self, a):
		"""Sum the log probabilities in an array.

		@param a: array logs
		@type a: array of float
		@return: log(exp(a[0]) + exp(a[1]) + ... + exp(a[n]))
		@rtype: float
		"""
		m = array([-inf])
		for element in a[nonzero(a != -inf)]:
			logaddexp(m,element,m)
		return m[0]


def multinomial_sample(distribution):
	"""Sample a random integer according to a multinomial distribution.

	@param distribution: probabilitiy distribution
	@type distribution: array of log probabilities
	@return: integer in the range 0 to the length of distribution
	@rtype: integer
	"""
	return multinomial(1, exp(distribution)).argmax()


def generate_corpus(categories, vocabulary, documents,
					hyp_pi = None, hyp_thetas = None):
	"""Create true parameters and sample data for a corpus of labeled documents.

	@param categories: number of categories
	@type categories: integer
	@param vocabulary: vocabulary size
	@type vocabulary: integer
	@param documents: number of documents in the corpus
	@type documents: integer
	@param hyp_pi: optional category hyperparamter, default uninformative
	@type hyp_pi: list or None
	@param hyp_thetas: optional word count hyperparamter, default uninformative
	@type hyp_thetas: list or None
	@return: word distributions per category, documents, document labels
	@rtype: tuple
	"""
	# Set up the hyperparameters.
	if hyp_pi == None:
		hyp_pi = [1]*categories
	if len(hyp_pi) != categories:
		raise Exception()
	if hyp_thetas == None:
		hyp_thetas = [1]*vocabulary
	if len(hyp_thetas) != vocabulary:
		raise Exception()
	# Generate the true model parameters.
	pi = log(dirichlet(hyp_pi, 1)[0])
	thetas = dirichlet(hyp_thetas, categories)
	# Generate the corpus and the true labels.
	corpus = empty((documents, vocabulary), int)
	labels = empty(documents, int)
	for document_index in xrange(documents):
		category = multinomial_sample(pi)
		labels[document_index] = category
		corpus[document_index] = multinomial(vocabulary*100, thetas[category])
	return log(thetas), corpus, labels


if __name__ == "__main__":
	# Generate data set
	categories = 10	# number of categories
	vocabulary = 5	# vocabulary size
	documents = 10	# dataset size

	true_theta, corpus, true_labels = \
		generate_corpus(categories, vocabulary, documents)
	print "true thetas\n%s" % true_theta
	print "true labels %s" % true_labels
	print "corpus\n%s" % corpus

	# Create the Gibbs sampler.
	hyp_pi = ones(categories, int)						# uninformed label prior
	hyp_thetas = ones((categories, vocabulary), int)	# uninformed word prior
	sampler = GibbsSampler(hyp_pi, hyp_thetas, corpus)

	# Use the sampler to estimate the document labels.
	estimates = sampler.estimate_labels(20, 5, 2)
	print "\nEstimated labels\n%s" % estimates
