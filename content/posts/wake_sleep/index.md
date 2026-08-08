---
title: "The wake-sleep algorithm"
date: 2026-08-08
draft: true
bib:
  - id: "hinton1995wake"
    title: "The \"Wake-Sleep\" Algorithm for Unsupervised Neural Networks"
    author: "Hinton, Geoffrey E and Dayan, Peter and Frey, Brendan J and Neal, Radford M"
    year: 1995
    journal: "Science, Vol 268, pp. 1158-1161"
    url: "https://www.cs.toronto.edu/~hinton/absps/ws.pdf"
---

While reading about self-supervised approaches to inverse problem for my pHd, I have encouter an algorithm I was not aware of: the wake-sleep algorithm ({{< citep "hinton1995wake" >}}).
In my opinion, this paper turns out to be one of the most pionneer work I have ever encounter:
the name "The \"Wake-Sleep\" Algorithm for Unsupervised Neural Networks" was released ... in 1995!
With this blogpost, my goal is to detail my understanding of this method and connect to modern tools, so that it the credit I think it deserves. 

# Notations
Consider the setup of inverse problems:
we would like to try to recover a quantity of interest $\vx$ from noisy indirect measurements $\vy$.
It is especially useful among scientific applications: tomography, microscopy, astronomy, geophysics, oceanography, climate modeling, and plenty of others.

The maths relies on Bayesian statistics --- where distributions capture uncertainty about quantities --- and borrow most of its terminology:
- roman synbol $a$ is a scalar, bold $\va$ is a vector, $\mA$ a matrix.
- font $\ra$ is a random scalar variable, bold $\rva$ is a random vector
- $\vy$ is measurement, $\rvy$ is the set of all measurements, linked to the data $\pdata(\rvy)$.
- $\vx$ is a solution, $\rvx$ is the random variable linked to the prior $p(\rvx)$ 
- $p(\vx)$ is the prior, which capture previous knowledge about the world of emitter. $p(\vy \mid \vx)$ is the likelihood, that shows how a measurement degrades the initial quantity.
- $p(\vy)$ is the evidence (also sometime called marginal likelihood). $p(\vx \mid \vy)$ is the posterior, or the information a posterior we have thanks to a measurement

# Bayes' rule

Everything start from probably one of probabilities' most importan theorem:
$$ p(\vx) p(\vy \mid \vx) = p(\vx, \vy) = p(\vy) p(\vx \mid \vy).$$
It captures the definition of probability theory, and connects all the pieces we have seen previously.
The left part represent the forward process, which follow the causal flow of time: we start by sampling an element, then this element is degraded through a process to produce $y$.
By contrast, the right side of the equation does the revser: it goes back in time, first getting a measurement and then infering from it the underlying quantity $\vx$. 

We also do hypothesis on the formulation process: a very simple for would be $\vy = \mH \vx + \vepsilon$ where $\mH$ is a linear degradation kernel and $\vepsilon$ is some gaussian noise.
So if we known those parameters, the degradation operator and the level $\sigma$ of the noise, then we have access to the likelihood, which we write $p_\phi(\vy \mid \vx)$ where $\phi = (\mH, \sigma)$.
The prior $p(\vx)$ is typically harder to get: it represents the prior knowledge we have about what we want to estimate.
For example, if we take an MRI of a human body, then we known ahead of time that we will encouter tissues, bones and blood.
Everything we known in advance, we can put it here.
Let's name the forward time process $P_\phi(\vx, \vy) = p_\phi(\vx) p_\phi(\vy \mid \vx)$.

Conversely, we already say that we now the evidence we generally know $p(\vy)$, wich we replace by $\pdata(\vy)$, our dataset of measurements.
Howveer, the posterior $p(\vx \mid \vy)$ is generally the difficult one: from an estimate, do we have a strategy to recover the distribution of possible causes that may have casued it?

# Learn the posterior, or the sleep phase.
## Link with NPE

# Learn a generator, or the wake phase.
## Link with VAE

# Alternate and converge
## Link with the EM algorithm
