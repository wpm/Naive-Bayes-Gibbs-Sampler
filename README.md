Naive Bayes Gibbs Sampler
=========================

This project implements the Gibbs sampler for Naive Bayes document classification described in [Resnik and Hardisty 2010, "Gibbs Sampling for the Uninitiated"](http://drum.lib.umd.edu/handle/1903/10058).
It closely follows the notation and design put forth in that paper.

There are only two significant differences between the algorithm presented in the paper and the code here.
Unlike the paper, this code allows the documents to be grouped into more than two classes, so document priors
are generated using a Dirichlet distribution rather than a Beta distribution.
Also the equations in the paper deal with probabilities, whereas to avoid underflow errors this code works with log probabilities. 

This code depends on [Numeric Python](http://numpy.scipy.org/).
