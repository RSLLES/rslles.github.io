---
title: "Maximum mean discrepancy"
date: 2026-03-02
lastmod: 2026-05-24
draft: false
bib:
  - id: "aronszajn1950theory"
    title: "Theory of reproducing kernels"
    author: "Aronszajn, Nachman"
    year: 1950
    journal: "Transactions of the American mathematical society"
    url: "https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/S0002-9947-1950-0051437-7.pdf"
  - id: "dudley2002real"
    title: "Real analysis and probability"
    author: "Dudley, Richard M"
    year: 2002
    journal: "Cambridge University Press"
    url: "https://assets.cambridge.org/97805218/09726/sample/9780521809726ws.pdf"
  - id: "rahimi2007random"
    title: "Random features for large-scale kernel machines"
    author: "Rahimi, Ali and Recht, Benjamin"
    journal: "{{< venue NeurIPS >}}"
    year: 2007
    url: "https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf"
  - id: "gretton2012kernel"
    title: "A Kernel Two-Sample Test"
    author: "Gretton, Arthur and Borgwardt, Karsten M and Rasch, Malte J and Scholkopf, Bernhard and Smola, Alexander"
    year: 2012
    journal: "{{< venue JMLR >}} 13, 723-773"
    url: "https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf"
  - id: "mairal2022kernel"
    title: "Kernel methods for machine learning"
    author: "Mairal, Julien and Arbel, Michael and Vert, Jean-Philippe"
    year: 2022
    journal: "Course for MVA Master's students at ENS Paris Saclay"
    url: "https://lear.inrialpes.fr/people/mairal/classes.html"
  - id: "schrab2023mmd"
    title: "MMD Aggregated Two-Sample Test"
    author: "Schrab, Antonin and Kim, Ilmun and Albert, Melisande and Laurent, Beatrice and Guedj, Benjamin and Gretton, Arthur"
    year: 2023
    journal: "{{< venue JMLR >}}"
    url: "https://www.jmlr.org/papers/volume24/21-1289/21-1289.pdf"
  - id: "biggs2023mmd"
    title: "MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting"
    author: "Biggs, Felix and Schrab, Antonin and Gretton, Arthur"
    year: 2023
    journal: "{{< venue NeurIPS >}}"
    url: "https://proceedings.neurips.cc/paper_files/paper/2023/file/edd00cead3425393baf13004de993017-Paper-Conference.pdf"
  - id: "hertrich2023generative"
    title: "Generative sliced MMD flows with Riesz kernels"
    author: "Hertrich, Johannes and Wald, Christian and Altekruger, Fabian and Hagemann, Paul"
    year: 2024
    journal: "{{< venue ICLR >}}"
    url: "https://arxiv.org/abs/2305.11463"
  - id: "mukherjee2025minimax"
    title: "Minimax Optimal Kernel Two-Sample Tests with Random Features"
    author: "Mukherjee, Soumya and Sriperumbudur, Bharath K"
    journal: "{{< venue arXiv >}}"
    year: 2025
    url: "https://arxiv.org/abs/2502.20755"
  - id: "balestriero2025lejepa"
    title: "Lejepa: Provable and scalable self-supervised learning without the heuristics"
    author: "Balestriero, Randall and LeCun, Yann"
    journal: "{{< venue arXiv >}}"
    year: 2025
    url: "https://arxiv.org/abs/2511.08544"
---

Statistics is largely about understanding and reasoning with distributions. 
In this domain, a common objective is to measure how two distributions compare to each other;
you can think about the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 
and all [integral probability metric](https://en.wikipedia.org/wiki/Integral_probability_metric) such as the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) or the [total variation distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) for instance --- but they generally require direct access to the probability distribution functions (p.d.f).

In contrast, in machine learning and especially when dealing with distributions over high-dimensional spaces, we often only have access to samples of the distribution, and not to the p.d.f themselves.
Imagine for example a dataset of measures from a phenomenon you want to model.
If you want to compare your model to the experimental observed measures, a strategy could be to sample $N$ data points from your model,
and then to use a tool capable of comparing those distributions solely based on their samples.
This is precisely the goal of the Maximum Mean Discrepancy (MMD) ({{< citep "gretton2012kernel" >}}), which is the topic of today's post.

## Intuition and a practical example
**The MMD is a technique to compare samples and verify whether they come from the same distribution or not.** 
Formally, it answers **the two-sample test**, that is a statistical test where the null hypothesis is that the two samples come from the same distribution.

In practice, if you have $m$ samples $\mX = (\vx_1, \dots, \vx_m)$ from one distribution $p$ and $n$ samples $\mY = (\vy_1, \dots, \vy_n)$ from another distribution $q$, then the MMD between $\mX$ and $\mY$ will transcribe how close the distributions $p$ and $q$ are from each other.

Let's have a look at the expression of the MMD.
Given the previous samples $\mX$ and $\mY$ respectively from $p$ and $q$, we also define $k: \gX \times \gX \longmapsto \R$ to be a **positive definite kernel**, i.e. a two variables function that is *symmetric and positive semidefinite* ({{< citep "aronszajn1950theory" >}}).
Then the MMD writes:
$$
\begin{split}
\hat{\operatorname{MMD}}^2[\mX, \mY] = \;
&\frac{1}{m(m-1)}\sum_{1\leq i,j \leq m,j\neq i} k(\vx_i,\vx_j) + \frac{1}{n(n-1)}\sum_{1\leq i,j \leq n,j\neq i} k(\vy_i,\vy_j)\\
&- \frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^n k(\vx_i,\vy_j).
\end{split}
$$

The kernel function can be thought of as a similarity function that measures how alike two elements are.
In the expression of the MMD, the first two terms measure the average self-similarity within each distribution, i.e. how alike samples from the same distribution tend to be.
The last term is a cross term that measures similarity across the two distributions.
When $\mX = \mY$, all three terms cancel out and the MMD is null.
We redirect the curious reader to the excellent course of {{< citet "mairal2022kernel" >}} for additional information about kernel methods for machine learning.

In machine learning, a common usage of MMD is to train **generative models**. 
Indeed, if you have access to a dataset, we can call this dataset $\mX = (\vx_1, \dots, \vx_m)$ and hypothesize that they all come from a higher distribution $p$ that you would like to sample.
Then if we define a new distribution $q_\theta$ using a neural network from which we can easily sample from, we can use the MMD as a loss function to bring $q_\theta$ close to $p$ !
Here is an illustration with a toy example where we train a generator to sample from a simple spiral density:

![Figure 1](./intro.avif)

Code and discussion around this example is available in the last section of this blog.

## The theory


Let's start with a basic characterisation in statistics.
Let $\rvx$ be a random variable and $p$ a distribution on a topological space $\gX$.
The [law of the unconscious statistician](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician) (or [*théorème de transfert*](https://www.bibmath.net/dico/index.php?action=affiche&quoi=.%2Ft%2Ftransfert.html) in French) says:

$$\rvx \sim p \Rightarrow \forall h \in \gC(\gX), \E_{\rvx \sim p}[h(\rvx)] = \int_\gX h(\vx) p(\vx) d\vx,$$

where $\gC(\gX)$ is the space of bounded continuous functions with values in $\mathbb{R}$.
What is really of interest is that the reverse implication of this property holds ({{< citep "dudley2002real" >}}), providing a characterization of probability measures. 
If $\rvx \sim p$ and $\rvy \sim q$, then:
$$p=q \Leftrightarrow  \forall h \in \gC(\gX), \E_{\rvx \sim p}[h(\rvx)] = \mathbb E_{\rvy \sim q}[h(\rvy)].$$

Using this characterization, we can see that if $\rvx$ and $\rvy$ do not follow the same distribution, then there must exist a function $h$ such that the expected values of $h(\rvx)$ and $h(\rvy)$ are different.
Naturally, we expect that the more different $\rvx$ and $\rvy$, the easier it should be to find a function $h$ that exacerbates their difference.

The fundamental idea of {{< citet "gretton2012kernel" >}} relies on this insight: they define **the MMD as the greatest possible difference between the expected values of $h(\rvx)$ and $h(\rvy)$ for a function $h$ in $\gH$ that maximizes this precise quantity**. In mathematical terms:

$$\operatorname{MMD}[\gH, p, q] = \sup_{h \in \mathcal \gH} \left( \E_{\rvx \sim p}[h(\rvx)] - \E_{\rvy \sim q}[h(\rvy)]  \right).$$

Of course, finding a supremum over a space of functions is far from trivial.
But this apparently complex optimization problem has a closed form for a very specific class of functions: when $h$ is restricted to the unit ball of a reproducing kernel Hilbert space (RKHS) ({{< citep "aronszajn1950theory" >}}) $\gH$.

Let's pick a RKHS $\gH$.
By the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem),
we denote $\phi: \gX \longmapsto \gH$ the feature mapping of $\gH$ and the kernel $k(\vx, \vy) = \dotprod{\phi(\vx)}{\phi(\vy)}_\gH$ for $\vx, \vy \in \gX^2$.

Let $h$ be a function of $\gH$.
For all $\vx \in \gX$, we have $h(\vx) = \dotprod{\phi(\vx)}{h}_\gH$.
We can use this property to derive a formula for $\E_{\rvx \sim p}[h(\rvx)]$:

$$
\begin{split}
\E_{\rvx \sim p} (h(\rvx)) &= \int_{\vx \in \gX}{h(\vx)p(\vx)dx} = \int_{\vx \in \gX}{\dotprod{\phi(\vx)}{h}_\gH p(\vx)dx} \\
&=  \dotprod{\int_{\vx \in \gX} \phi(\vx) p(\vx)dx}{h}_\gH = \dotprod{\mu_p}{h}_\gH,
\end{split}
$$

where $\mu_p \in \gH$ is the expected value of the random feature mapping $\phi(\rvx)$.
It can be seen as an element of the RKHS that embeds the distribution $p$, as it allows evaluating the expected value of the distribution against any arbitrary function $h$ by a simple dot product.

This expression is extremely convenient, as we can now simplify the difference we wish to calculate:

$$\E_{\rvx \sim p}[h(\rvx)] - \E_{\rvy \sim q}[h(\rvy)] = \dotprod{\mu_p - \mu_q}{h}_\gH.$$

Given the linearity of this expression, the supremum becomes obvious to calculate.
To prevent an infinite result, we can restrict $\gH$ to the unit hyper-ball of the RKHS,
i.e. working with $\gH' = \{h \in \mathcal \gH \; \text{such that} \; ||h||_\gH \leq 1\}$.
Then, this quantity is maximized for $h=\frac{\mu_p - \mu_q}{||\mu_p - \mu_q||_\gH}$,
that is the vector of norm 1 pointing in the exact direction of $\mu_p - \mu_q$.
The formula to compute the MMD is then:

$$
\begin{split}
\operatorname{MMD}[\gH', p, q]
&= \sup_{\{h \in \mathcal \gH \; \text{such that} \; ||h||_\gH \leq 1\}} \left( \E_{\rvx \sim p}[h(\rvx)] - \E_{\rvy \sim q}[h(\rvy)]  \right) \\
&= \dotprod{\mu_p - \mu_q}{\frac{\mu_p - \mu_q}{||\mu_p - \mu_q||_\gH}}_\gH \\
&= ||\mu_p - \mu_q||_\gH.
\end{split}
$$

{{< citet "gretton2012kernel" >}} have shown that the MMD is a metric, in the sense defined by {{< citet "dudley2002real" >}}, i.e. it satisfies symmetry, the triangle inequality, positivity, and $\operatorname{MMD}[\gH', p, q] = 0 \Rightarrow p=q$.

The last step is to find a way to compute this quantity.
$\mu_p$ and $\mu_q$ are elements of $\gH$, i.e. functions, and are intractable as such.
The whole philosophy of kernel methods in general is to use these high-level objects
to prove some theory, but in the end to get back to simple expressions directly involving evaluation of the kernel $k: \gX \times \gX \longmapsto \R$, which is tractable.

We can develop the expression of the MMD squared:

$$
\begin{split}
\operatorname{MMD}^2[\gH', p, q]
&= ||\mu_p - \mu_q||^2_\gH = \dotprod{\mu_p - \mu_q}{\mu_p - \mu_q}_\gH \\
&= \dotprod{\mu_p}{\mu_p}_\gH -2\dotprod{\mu_p}{\mu_q}_\gH + \dotprod{\mu_q}{\mu_q}_\gH.
\end{split}
$$

These scalar products between distribution embeddings can be expressed using the kernel.
Say we have two independent random variables $\rva \sim \rho_1$ and $\rvb \sim \rho_2$, then:

$$
\begin{split}
\E_{\rva \sim \rho_1, \rvb \sim \rho_2}[k(\rva, \rvb)]
&= \E_{\rva \sim \rho_1, \rvb \sim \rho_2}[\dotprod{\phi(\rva)}{\phi(\rvb)}_\gH] \\
&= \dotprod{\E_{\rva \sim \rho_1}[\phi(\rva)]}{\E_{\rvb \sim \rho_2}[\phi(\rvb)]}_\gH \\
&= \dotprod{\mu_{\rho_1}}{\mu_{\rho_2}}_\gH.
\end{split}
$$

Therefore, we have:
$$
\begin{split}
\operatorname{MMD}^2[\gH', p, q]
&= \dotprod{\mu_p}{\mu_p}_\gH -2\dotprod{\mu_p}{\mu_q}_\gH + \dotprod{\mu_q}{\mu_q}_\gH \\
&= \E_{\rvx_1 \sim p, \rvx_2 \sim p}[k(\rvx_1, \rvx_2)] -2\E_{\rvx \sim p, \rvy \sim q}[k(\rvx, \rvy)] + \E_{\rvy_1 \sim q, \rvy_2 \sim q}[k(\rvy_1, \rvy_2)].
\end{split}
$$

If we assume that we only have access to samples from both distributions, say $m$ samples $\mX = (\vx_1, \dots, \vx_m)$ from $p$ and $n$ samples $\mY = (\vy_1, \dots, \vy_n)$ from $q$,
we can use those to estimate the expected values.
Replacing these by their Monte-Carlo estimators yields an unbiased estimator for the MMD squared:

$$
\begin{split}
\hat{\operatorname{MMD}}^2[\mX, \mY] = \;
&\frac{1}{m(m-1)}\sum_{1\leq i,j \leq m,j\neq i} k(\vx_i,\vx_j) \\
&- \frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^n k(\vx_i,\vy_j) \\
&+ \frac{1}{n(n-1)}\sum_{1\leq i,j \leq n,j\neq i} k(\vy_i,\vy_j).
\end{split}
$$

We have derived the MMD formula from scratch, defining a metric that can assess whether two distributions are alike based solely on samples from both.

A last adjustment should be made though.
Because we have used Monte Carlo estimators, nothing prevents this quantity from being negative.
Therefore, a traditional adjustment is to slightly bias the MMD by including the diagonal terms, creating a new estimator that we denote as $\hat{\operatorname{MMD}}_\text{biased}$ that guarantees a non-negative result:

$$
\begin{split}
\hat{\operatorname{MMD}}_\text{biased}^2[\mX, \mY] = \;
&\frac{1}{m^2}\sum_{1\leq i,j \leq m} k(\vx_i,\vx_j) \\
&- \frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^n k(\vx_i,\vy_j) \\
&+ \frac{1}{n^2}\sum_{1\leq i,j \leq n} k(\vy_i,\vy_j).
\end{split}
$$

For $m$ and $n$ sufficiently large, this should not impact the MMD too much.

## Reduce the complexity

The previous derived expression is not suited for online usage.
Indeed, for any new point added to the dataset, we need to compute its application through the kernel against all previous points,
meaning we need to keep the entire history, and have a quadratic scaling with respect to the number of samples.
Below are two methods to downscale the complexity to $\gO(n)$: 
the **linear MMD** approximation, which reduces complexity at the cost of a bigger variance,
and a special identity with **stationary kernels**. 

### Linear MMD

If we look at the previous formula, we see that **the minimum number of samples required by the MMD is 2 from each distribution**.
Given only $\vx_1, \vx_2$ from $p$ and $\vy_1, \vy_2$ from $q$, we can approximate
$\E_{\rvx_1 \sim p, \rvx_2 \sim p}[k(\rvx_1, \rvx_2)]$ by $k(\vx_1, \vx_2)$,
$\E_{\rvy_1 \sim q, \rvy_2 \sim q}[k(\rvy_1, \rvy_2)]$ by $k(\vy_1, \vy_2)$,
and $\E_{\rvx \sim p, \rvy \sim q}[k(\rvx, \rvy)]$ by $\frac{1}{2}(k(\vx_1, \vy_2) + k(\vx_2, \vy_1))$.
This yields the lightest possible estimator for the squared MMD, 
that we denote as $\hat{\operatorname{MMD}}^2_2$ and is:

$$
\hat{\operatorname{MMD}}^2_2[\vx_1, \vx_2, \vy_1, \vy_2] = k(\vx_1, \vx_2) + k(\vy_1, \vy_2) - k(\vx_1, \vy_2) - k(\vx_2, \vy_1).
$$

This estimate is of course highly noisy --- being based on two samples --- 
but is also extremely cheap to compute.
**Linear MMD leverages this formulation by computing online $\hat{\operatorname{MMD}}^2_2$ for each new pair of points, and averages their value over time to create a linear estimator.**
Assuming we have $2n$ samples $(\vx_1, ..., \vx_{2n})$ from $p$ and also $2n$ samples $(\vy_1, ..., \vy_{2n})$ from $q$, we can define the linear MMD estimator as:

$$
\begin{split}
\hat{\operatorname{MMD}}_\text{linear}^2[\mX, \mY] 
&= \frac{1}{n}\sum_{i=1}^n \hat{\operatorname{MMD}}^2_2[\vx_{2i-1}, \vx_{2i}, \vy_{2i-1}, \vy_{2i}] \\
&= \frac{1}{n}\sum_{i=1}^n \left[ k(\vx_{2i-1}, \vx_{2i}) + k(\vy_{2i-1}, \vy_{2i}) - k(\vx_{2i-1}, \vy_{2i}) - k(\vx_{2i}, \vy_{2i-1}) \right].
\end{split}
$$

This formulation enables online MMD estimation: it requires $\gO(1)$ memory cost and $\gO(n)$ computational cost, to the detriment of higher variance.

With large datasets in machine learning, we may also encounter scenarios where the total number of samples is too high to compute the quadratic classic MMD, but a higher-order MMD than the simple $\hat{\operatorname{MMD}}_2$ can be computed.
You can generalize the previous idea by fixing a block size $b$ specific to your compute budget, and use the following estimator:

$$
\hat{\operatorname{MMD}}_{b\text{-linear}}^2[\mX, \mY] 
= \frac{1}{n}\sum_{i=1}^n \hat{\operatorname{MMD}}^2_{b}[(\vx_{ib+1}, \dots, \vx_{(i+1)b}), (\vy_{ib+1}, \dots, \vy_{(i+1)b})] \\
$$

### Simplification for stationary kernels

A kernel $k(\vx, \vy)$ is stationary if it depends only on the difference $\vu = \vx - \vy$, i.e. $k(\vx, \vy) = \psi(\vx - \vy)$.
Among others, it includes the radial basis function kernel (RBF), $k_\text{RBF}(\vx, \vy) = \exp \left(- \frac{1}{2} ||\vx - \vy ||_2^2 / \sigma^2 \right)$ and the Laplace kernel, $k_\text{L}(\vx, \vy) = \exp \left(- \gamma ||\vx - \vy ||_1 \right)$.

If you are familiar with the theory of linear shift-invariant systems, you may see where this is going.
Because these kernels are translation invariant, **they are convolution kernels**, and then it may be a good idea to look at their Fourier transform.

And indeed, a key result is [Bochner's theorem](https://en.wikipedia.org/wiki/Bochner's_theorem): an stationary kernel $\psi$ is s.p.d iff $\psi$ is the Fourier transform of a non negative measure spectral measure $\nu$:

$$k(\vx, \vy) = \psi (\vx - \vy) = \int_{\R^d} \exp \left( i \vw^T (\vx - \vy) \right) \nu(\vw) d\vw $$

Why is it convenient ? Well, this last integral is nothing more than a scalar product:

$$ 
\begin{split}
k(\vx, \vy) &= \int_{\R^d} \exp \left( i \vw^T (\vx - \vy) \right) \nu(\vw) d\vw \\
&= \int_{\R^d} \exp \left( i \vw^T \vx \right) \overline{\exp \left(i \vw^T \vy \right)} \nu(\vw) d\vw \\
&= \dotprod{\phi(\vx)}{\phi(\vy)}_\gH,
\end{split}
$$

where the feature map $\phi(\vx)$ is a function from $\R^d$ to $\sC$ such that $\phi(\vx): \vw \longmapsto e^{i \vw^T \vx}$,
and the dot product $\dotprod{.}{.}_\gH$ is the canonical dot product of two functions $a$ and $b$ with values in $\sC$ and for a measure $\nu$: $\dotprod{a}{b}_\gH = \int_{\R^d} a(\vw) \overline{b(\vw)} \nu(\vw) d\vw$.

Hence, with Bochner's theorem, we have found the analytic expression of the decomposition of the kernel according to the Riesz representation theorem.
This is convenient: we can now compute each feature vector independently, e.g. over $p$:

$$ \mu_p(\vw) = \int_{\vx \in \gX} [\phi(\vx)](\vw) p(\vx)d\vx 
=  \int_{\vx \in \gX} e^{i \vw^T \vx} p(\vx)d\vx
= \E_{\vx \sim p} \left[ e^{i \vw^T \vx} \right].
$$

For readers familiar with Fourier analysis, this quantity is precisely the [characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)) of $p$, evaluated at $\vw$.
In other words, **for a stationary kernel, the feature map will always be the characteristic function of the random variable.**

Finally, given that we choose a normalized kernel, $\nu$ will be a probability measure, and thus we can re-write the MMD as an expectation:

$$ \begin{split}
\operatorname{MMD}^2(\gH, p, q) &= ||\mu_p - \mu_q ||_\gH^2 
= \int_{\R^d} | \mu_p(\vw) - \mu_q(\vw) |^2 \; \nu(\vw) d\vw \\ 
&= \E_{\vw \sim \nu} \left[ | \mu_p(\vw) - \mu_q(\vw) |^2 \right].
\end{split}$$

TLDR, **the specific structure of the kernel allows to directly compute the distribution embeddings instead of relying on the developed form of the MMD**.
And because all elements are expectations, we can approximate them using Monte-Carlo sums !

The approximation by MC sums is exactly the idea of Random Fourier Features ({{< citep "rahimi2007random" >}}) applied to the MMD (RFF-MMD) ({{< citep "mukherjee2025minimax" >}}):

$$\hat{\operatorname{MMD}}_\text{RFF}^2(X, Y)= \frac{1}{r} \sum_{k=1}^r |\hat \mu_p^{(k)} - \hat \mu_q^{(k)}|^2
\; \text{where} \; 
\begin{cases}
\vw_1, \dots, \vw_r \overset{\text{i.i.d}}{\sim} \nu &,\\
\hat \mu_p^{(k)} = \frac{1}{m}\sum_{j=1}^m e^{i \vw_k^T \vx_j} &,\\
\hat \mu_q^{(k)} = \frac{1}{n}\sum_{j=1}^n e^{i \vw_k^T \vy_j} &.\\
\end{cases} $$

As you can see, there is no need to compute any $n \times m$ matrix: we have effectively reduced the complexity to $\gO(n)$ (assuming $m=n$).

This idea has been explored in numerous works: and {{< citet "balestriero2025lejepa" >}} use this algorithm to train the latest JEPA at the time of writing. 

Finally, here is a table of the distribution $\nu$ for the two most popular stationary kernels:


| Kernel  | Formula                                   | Probability dist. $\nu(\vw)$                                            | Notes                                |
| :------ | :---------------------------------------- | :---------------------------------------------------------------------- | ------------------------------------ |
| RBF     | $e^{-\frac{1}{2\sigma^2}\|\vx-\vy\|_2^2}$ | $\vw \sim \mathcal{N}(\mathbf{0},\, \sigma^{-2} I_d)$                   | Isotropic normal with std $1/\sigma$ |
| Laplace | $e^{-\gamma\|\vx-\vy\|_1}$                | $\vw = (w_1, \dots, w_d)$ where $w_j \sim \mathrm{Cauchy}(0,\, \gamma)$ | Each component of $\vw$ is i.i.d     |


## Example usage

Below is an example of using MMD as a loss function for a toy generative task.
The code is open source and available [here](https://github.com/RSLLES/pytorchblueprint), along with other generative methods.
The goal is to generate samples from the following spiral distribution:

![Target spiral distribution](./target_dist.avif)

We define a 2-layer MLP neural network that takes 2D input and produces 2D outputs:
```python
class FourierEmbeddings(nn.Module):
    """Project data to sin and cosine embeddings with normaly sampled weights."""

    def __init__(self, input_dim: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        assert embed_dim % 2 == 0
        angular_freqs = scale * 2 * torch.pi * torch.randn(input_dim, embed_dim // 2)
        self.register_buffer("angular_freqs", angular_freqs)

    def forward(self, t: Tensor) -> Tensor:  # noqa: D102
        angles = t @ self.angular_freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class ResnetBlock(nn.Module):
    """ResNet block with SiLU activation."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()

    def forward(self, x):  # noqa: D102
        return self.act(x + self.block(x))


class MomentNet(nn.Module):
    """Toy model that learns a input_dim-D  function."""

    def __init__(self, input_dim: int, inner_dim: int, embed_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            FourierEmbeddings(input_dim, inner_dim, scale=embed_scale),
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            ResnetBlock(inner_dim),
            ResnetBlock(inner_dim),
            nn.Linear(inner_dim, input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        outputs_flat = self.net(x_flat)
        outputs = outputs_flat.view(*x_shape)
        return outputs
```

We define the following training loop, 
that uses the model to send elements from one source distribution (here a simple 2D gaussian) to mimic the target distributions (the spiral):
```python
class MomentMatchingTrainer(nn.Module):
    def __init__(self, model: nn.Module, kernel: nn.Module):
        super().__init__()
        self.model = model
        self.loss_func = MMD(kernel=kernel)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_source = x["sourcedist"]
        x_target = x["targetdist"]
        x_pred = self.model(x_source)
        loss = self.loss_func(x_pred[None], x_target[None])
        return loss
``` 

Here is the MMD code:
```python
def mmd(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    """Return the (biased) MMD between x and y given a kernel."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    if x.size(1) < 2 or y.size(1) < 2:
        raise ValueError(
            f"MMD needs at least 2 samples per batch, got x:{x.size(1)}, y:{y.size(1)}."
        )

    N = x.size(1)
    xy = torch.cat([x, y], dim=1)
    K = kernel(xy, xy)

    K_xx = K[:, :N, :N].mean(dim=(1, 2))
    K_yy = K[:, N:, N:].mean(dim=(1, 2))
    K_xy = K[:, :N, N:].mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy
    return reduce(mmd_sq, dim=0, mode=reduction)
    
class MMD(nn.Module):
    """Batchified MMD."""

    def __init__(self, kernel: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return mmd(x=x, y=y, kernel=self.kernel, reduction=self.reduction)
```

For now, we will simply use a Laplace kernel (`p=1`).
Regarding the parameter $\sigma$, a simple and natural choice is to take the median of the distance matrix ({{< citep "gretton2012kernel" >}}).

```python
class ExponentialKernel(nn.Module):
    """Kernel that includes both Laplace Kernel (p=1) and Gaussian Kernel (p=2)."""

    def __init__(self, p: float | int, eps: float = 1e-9):
        super().__init__()
        self.p = p
        self.eps = eps

    def compute_bandwidth(self, dists: Tensor) -> Tensor:
        """Return the median of the non-zero elements."""
        dists = dists.detach()
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        median = torch.nanquantile(dists_no_zeros, 0.5)
        return median

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:  # noqa: D102
        dists = torch.cdist(z1, z2, p=self.p).pow(self.p)
        kernel = torch.exp(-dists / self.compute_bandwidth(dists))
        return kernel
```

Here are the visual results, after $\sim$10,000 steps with a batch size of `1024`:
![MMD vanilla](./mmd_default.avif)

To provide a quantitative metric to assess the quality of the results, 
we propose to use the [earth mover's distance](https://en.wikipedia.org/wiki/Earth_mover's_distance) (EMD) between the sampled distribution "Results" and the target distribution "Target" as a validation metric.
With the previous distribution, the EMD is `0.168`. 

You may agree with me that the previous results are ... well, not that great.
This is because MMD is highly sensitive to the kernel you pick !
Here, while the laplace kernel is usually a safe bet, picking a simple median estimate for the bandwidth is not precise enough.
Therefore, the next section is about the latest bandwidth selection methods I could find at the time of writting to pick great kernerls.

## Kernel selection methods
The performance of the MMD varies highly depending on the choice of the kernel.
While the Gaussian kernel --- also known as radial basis function kernel (RBF) --- is almost ubiquitous with the Laplace kernel, selecting the correct sigma --- also known as the bandwidth --- is critical.

First, on bandwidth selection. Better than the median, {{< citet "schrab2023mmd" >}}
recommend to take 10 different bandwidths uniformly spaced between half the 5th-percentile and twice the 95th-percentile values.
As the mean of the positive definite kernels is a positive definite kernel, you can simply use the mean of all the kernels defined with these bandwidths:

```python
class ExponentialKernel(nn.Module):
    """Kernel that includes both Laplace Kernel (p=1) and Gaussian Kernel (p=2)."""

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(self, p: float | int, n_kernels: int = 10, eps: float = 1e-9):
        super().__init__()
        self.p = p
        self.eps = eps
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))

    def compute_bandwidths(self, dists: Tensor) -> Tensor:
        """Interpolates bandwidths between 0.5*q05 and 2.0*q95., see [3]."""
        dists = dists.detach()
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:  # noqa: D102
        dists = torch.cdist(z1, z2, p=self.p).pow(self.p)
        kernels = torch.exp(-dists[..., None] / self.compute_bandwidths(dists))
        return reduce(kernels, dim=-1, mode=reduction)
```

Here are the results with this new kernel, with a new best scoring validation metric of `0.164`:
![MMD mean](./mmd_mean.avif)

You may notice that this code re-computes the percentiles for each new batch.
A natural idea to leverage previously computed values is to use an exponential moving average, which will build smoother and more robust estimates: 

```python
class ExpMovingAverage(nn.Module):
    """Exponential moving average for a 1D Tensor."""

    mean: Tensor
    is_initialized: Tensor

    def __init__(self, lambd: float, n_values: int = 1) -> None:
        super().__init__()
        self.lambd = lambd
        self.register_buffer("mean", torch.zeros((n_values,)))
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.compiler.disable
    def _sync_values(self, x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.AVG)
        return x

    @torch.compiler.disable
    def _initialize_buffer(self, x: Tensor):
        x_synced = self._sync_values(x)
        self.mean.copy_(x_synced)
        self.is_initialized.fill_(True)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Update the current ema and return the new value."""
        if not self.is_initialized:
            self._initialize_buffer(x)
            return self.mean
        x_synced = self._sync_values(x)
        self.mean.mul_(1.0 - self.lambd).add_(x_synced, alpha=self.lambd)
        return self.mean


class ExponentialKernel(nn.Module):
    """Kernel that includes both Laplace Kernel (p=1) and Gaussian Kernel (p=2)."""

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(self, p: float | int, n_kernels: int = 5, eps: float = 1e-9):
        super().__init__()
        self.p = p
        self.eps = eps
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))
        self.ema_q05 = ExpMovingAverage(1e-3)
        self.ema_q95 = ExpMovingAverage(1e-3)

    def compute_bandwidths(self, dists: Tensor) -> Tensor:
        """Interpolates bandwidths between 0.5*q05 and 2.0*q95., see [3]."""
        dists = dists.detach()
        # classic masking resizes dists_no_zeros dynamically which crashes the compiler.
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        q05, q95 = self.ema_q05(q05), self.ema_q95(q95)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:  # noqa: D102
        dists = torch.cdist(z1, z2, p=self.p).pow(self.p)
        kernels = torch.exp(-dists[..., None] / self.compute_bandwidths(dists))
        return reduce(kernels, dim=-1, mode=reduction)
```

This new estimation technique brings the validation EMD to `0.152`.

In the previous derivation, the MMD can be defined as a supremum over functions $h$ in $\gH'$.
Therefore, instead of a mean, we could compute the MMD for all these kernels and at the last moment use the one that provides the highest MMD value,
aligning kernel selection with the MMD philosophy.
We refer to this strategy as `mmd_max`, and it can be implemented with the following code:

```python
def mmd_max(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    N = x.size(1)
    xy = torch.cat([x, y], dim=1)
    K = kernel(xy, xy, reduction="none")

    K_xx = K[:, :N, :N].mean(dim=(1, 2))
    K_yy = K[:, N:, N:].mean(dim=(1, 2))
    K_xy = K[:, :N, N:].mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy
    mmd_sq = mmd_sq.amax(dim=-1)
    return reduce(mmd_sq, dim=0, mode=reduction)
```

With this new strategy, we have reached a validation metric of `0.133`.
![MMD max](./mmd_max.avif)

MMD complexity is $\gO(d(m+n)^2)$ where $m$ and $n$ are the number of samples from each distribution and $d$ the dimension of the data ($d=2$ with our toy example).
As explained before, with stationary kernels, we can reduce the complexity to $\gO(dmr+dnr)$ by introducing $r$ samples from the probability distribution associated with the kernel through Bochner's theorem.
For a RBF kernel, it is an isotropic normal distribution, hence we can play with the following code:

```python
def rff_mmd(
    x: Tensor, y: Tensor, dist: nn.Module, n_features: int, reduction: str
) -> Tensor:
    """Implement RFF MDD."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    w = dist(x, y, n_features=n_features)
    phase_x = torch.einsum("bnd,rdk->bnrk", x, w)
    phase_y = torch.einsum("bnd,rdk->bnrk", y, w)
    mu_x = torch.exp(1j * phase_x).mean(dim=1)
    mu_y = torch.exp(1j * phase_y).mean(dim=1)
    mmd_rff = (mu_x - mu_y).abs().square().mean(dim=1)
    mmd_rff = mmd_rff.amax(dim=1)
    return reduce(mmd_rff, dim=0, mode=reduction)

class NormalDistribution(nn.Module):
    """Normal sampler with multi-bandwidth scaling for RFF features.

    Bandwidths follow the same quantile-EMA recipe as ``ExponentialKernel``,
    but the pairwise distances are estimated on a random subsample of size
    ``n_subsample`` per side instead of the full N x N matrix.
    """

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(
        self,
        n_kernels: int = 5,
        n_subsample: int = 128,
        ema_lambd: float = 1e-3,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.n_subsample = n_subsample
        self.eps = eps
        self.n_kernels = n_kernels
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))
        self.ema_q05 = ExpMovingAverage(ema_lambd)
        self.ema_q95 = ExpMovingAverage(ema_lambd)

    def _subsample(self, z: Tensor) -> Tensor:
        N = z.size(-2)
        n = min(self.n_subsample, N)
        idx = torch.randperm(N, device=z.device)[:n]
        return z.index_select(-2, idx)

    @torch.no_grad
    def compute_bandwidths(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Interpolate bandwidths between 0.5*q05 and 2.0*q95 on subsampled dists."""
        z1_sub = self._subsample(z1)
        z2_sub = self._subsample(z2)
        dists = torch.cdist(z1_sub, z2_sub, p=2.0)
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        q05, q95 = self.ema_q05(q05), self.ema_q95(q95)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, n_features: int) -> Tensor:
        """Sample RFF frequencies of shape ``[n_features, D, n_kernels]``."""
        D = z1.size(-1)
        bandwidths = self.compute_bandwidths(z1, z2)
        w = torch.randn(n_features, D, 1, device=z1.device, dtype=z1.dtype)
        return w / bandwidths
```

With this approach, we reach a validation metric of `0.128`, all with a linear approximation.


Alternatively, in scenarios where $d \gg 1$, computing the MMD can also become compute-intensive.
A natural idea is then to **project the data into a lower-dimensional subspace** --- similarly to sliced Wasserstein distance --- with an orthogonal matrix.
For this, you can either use a random projection matrix, or even better, find one that maximizes the MMD.

Here is a very hacky (and completely sub-optimal) implementation of this idea: search for the projection that maximizes the MMD.
In practice, instead of a pass-through trick that computes the loss two times, you should add a separate optimizer for `W`.

```python
class MaxSliceKernel(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, base_kernel: nn.Module):
        super().__init__()
        self.base_kernel = base_kernel
        W = nn.Linear(input_dim, proj_dim, bias=False)
        self.proj = torch.nn.utils.parametrizations.spectral_norm(W)

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:
        W = self.proj.weight
        e1 = F.linear(z1, W.detach())
        e2 = F.linear(z2, W.detach())
        loss = self.base_kernel(e1, e2, reduction=reduction)
        e1_k = F.linear(z1.detach(), W)
        e2_k = F.linear(z2.detach(), W)
        loss_k = self.base_kernel(e1_k, e2_k, reduction=reduction)
        return loss + (loss_k.detach() - loss_k)
```

If we pick `proj_dim = input_dim = 2`, we reach an EMD of `0.124`.

![MMD max](./mmd_maxslice.avif)

However, if we pick `proj_dim = 1`, then the validation metric is `1.11`: it does not work at all.
This makes sense here: the target distribution is circular, and no single 1D projection can capture it.
If we want to rely on cheap 1D projections, we may need many of them:

```python
class SliceKernel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_proj: int,
        base_kernel: nn.Module = ExponentialKernel(p=1),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_proj = n_proj
        self.base_kernel = base_kernel

    def _random_proj(self, device) -> Tensor:
        u = torch.randn((self.input_dim, self.n_proj), device=device)
        return F.normalize(u, dim=0, p=2)

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:  # noqa: D102
        B, N1, _ = z1.shape
        B, N2, _ = z2.shape
        u = self._random_proj(device=z1.device)
        e1 = (z1 @ u).transpose(1, 2).reshape(B * self.n_proj, N1, 1)
        e2 = (z2 @ u).transpose(1, 2).reshape(B * self.n_proj, N2, 1)
        kernels = self.base_kernel(e1, e2, reduction="none")
        kernels = kernels.reshape(B, self.n_proj, N1, N2, -1).mean(dim=1)
        return reduce(kernels, dim=-1, mode=reduction)
```

Note that this approach offers some guarantees when paired with the Riesz kernel: $K(\vx,\vy) = - \| \vx - \vy \|_2^r$ with $r \in (0,2)$.
{{< citet "hertrich2023generative" >}} proved that using random projections with a Riesz kernel is equivalent to the complete MMD with this kernel,
which is neat for high-dimensional problems.
Furthermore, in the special case of the Riesz kernel with `r=1`, a clever sorting trick reduces the MMD complexity from $\gO((n_1 + n_2)^2)$ to $\gO((n_1 + n_2) \log (n_1 + n_2))$, making it blazingly fast for a large number of high-dimensional objects.

{{< citet "schrab2023mmd" >}} also recommend to use both Laplace and RBF kernels with their bandwidths selection strategy:
```python
class LaplaceGaussianKernel(nn.Module):
    """Mixed kernel combining Laplace (L1) and Gaussian (squared L2)."""

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(self, n_kernels: int = 5, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))
        self.ema_l1_q05 = ExpMovingAverage(1e-3)
        self.ema_l1_q95 = ExpMovingAverage(1e-3)
        self.ema_l2_q05 = ExpMovingAverage(1e-3)
        self.ema_l2_q95 = ExpMovingAverage(1e-3)

    def _bandwidths(
        self, dists: Tensor, ema_q05: ExpMovingAverage, ema_q95: ExpMovingAverage
    ) -> Tensor:
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        q05, q95 = ema_q05(q05), ema_q95(q95)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:  # noqa: D102
        l1 = torch.cdist(z1, z2, p=1)
        l2_sq = torch.cdist(z1, z2, p=2).square()

        bw_l1 = self._bandwidths(l1.detach(), self.ema_l1_q05, self.ema_l1_q95)
        bw_l2 = self._bandwidths(l2_sq.detach(), self.ema_l2_q05, self.ema_l2_q95)

        k_laplace = torch.exp(-l1[..., None] / bw_l1)
        k_gaussian = torch.exp(-l2_sq[..., None] / bw_l2)
        kernels = torch.cat([k_laplace, k_gaussian], dim=-1)
        return reduce(kernels, dim=-1, mode=reduction)
```

This slightly improved the performances to `0.131`, though I am not sure this is really significant.


Building upon this idea, {{< citet "biggs2023mmd" >}} proposes a fusion mechanism that can be seen as a soft max, 
that kind of interpolates between taking a mean and a max, hoping to get the stability of the mean with the theoretical advantages of the max.

```python
def mmd_fuse(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    N = x.size(1)
    xy = torch.cat([x, y], dim=1)
    K = kernel(xy, xy, reduction="none")
    n_kernels = torch.tensor(K.size(-1), device=K.device, dtype=K.dtype)
    lambd = torch.sqrt(n_kernels * (n_kernels - 1.0))

    K_xx = K[:, :N, :N].mean(dim=(1, 2))
    K_yy = K[:, N:, N:].mean(dim=(1, 2))
    K_xy = K[:, :N, N:].mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy

    std = K.square().mean(dim=(1, 2)).sqrt()
    mmd_norm = mmd_sq / (std + 1e-9)
    mmd_fused = (torch.logsumexp(lambd * mmd_norm, dim=-1) - n_kernels.log()) / lambd
    return reduce(mmd_fused, dim=0, mode=reduction)
```

With this new aggregation mechanism, we reach a validation metric of `0.150`, so slightly worse than the `max` strategy.

![MMD Fuse](./mmd_fuse.avif)

Keep in mind that this is a toy example, which may not reflect the conclusion of some other real life scenarios.
