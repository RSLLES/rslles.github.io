---
title: "The KoLeo regularization"
date: 2026-02-20
draft: false
bib:
  - id: "kozachenko1987sample"
    title: "Sample Estimate of the Entropy of a Random Vector "
    author: "Kozachenko, L. F. and Leonenko, N. N."
    year: 1987
    journal: "Problemy Peredachi Informatsii"
    url: "https://www.mathnet.ru/links/345304e244f0d2ec3706a1ce00fda1bb/ppi797.pdf"
  - id: "beirlant1997nonparametric"
    title: "Nonparametric entropy estimation: an overview"
    author: "Beirlant, Jan and Dudewicz, Edward J and Gyorfi, Laszlo and Van der Meulen, Edward C"
    year: 1997
    journal: "International Journal of the Mathematical Statistics Sciences"
    url: "http://jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf"
  - id: "delattre2017kozachenko"
    title: "On the Kozachenko–Leonenko entropy estimator"
    author: "Delattre, Sylvain and Fournier, Nicolas"
    journal: "Journal of Statistical Planning and Inference"
    year: 2017
    url: "https://arxiv.org/abs/1602.07440"
  - id: "sablayrolles2018spreading"
    title: "Spreading vectors for similarity search"
    author: "Sablayrolles, Alexandre and Douze, Matthijs and Schmid, Cordelia and Jégou, Hervé"
    journal: "International Conference on Learning Representations (ICLR)"
    year: 2019
    url: "https://arxiv.org/abs/1806.03198"
  - id: "oquab2023dinov2"
    title: "DINOv2: Learning Robust Visual Features without Supervision"
    author: "Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others"
    journal: "Transactions on Machine Learning Research (TMLR)"
    year: 2024
    url: "https://arxiv.org/abs/2304.07193"
---

In manifold learning, a common desire is to cover the latent space as much as possible.
Indeed, imagine configuring a neural network to use a latent space of dimension 3,
but after training, you realize that it effectively only uses a subset of it, say 1 dimension:
it feels like wasted resources.
In self-supervised representation works, such as DinoV2 ({{< citep "oquab2023dinov2" >}}), the goal is to provide meaningful embeddings to natural images.
Therefore, one would expect that given independent random natural images, the resulting DinoV2-embeddings should spread uniformly over the d-dimensional hyperball (that is the manifold of DinoV2's embeddings).

Well, that is precisely the mission of the *KoLeo regularization* - a regularization that encourages the uniform spreading of embeddings -, the topic of today's post.

## Formula and intuition
Consider $N$ vectors $\vx_1, \dots, \vx_n$ of $\R^d$.
The *KoLeo regularizer* defined by {{< citet "sablayrolles2018spreading" >}} writes:

$$\mathcal L_\text{KoLeo} = -\sum_{i=1}^N \log \min_{j \neq i} || \vx_i - \vx_j ||.$$

It corresponds to the **average negative log-distance of each point to its nearest neighbour**.
Therefore, minimizing $\mathcal L_\text{KoLeo}$ effectively maximizes the quantities $|| \vx_i - \vx_j ||$, repulsing vectors away from one another with a force that is decreasing with respect to their individual distance to their closest neighbour.
If we ignore the logarithm for a second, one may think of it as collection of compressed springs connecting pairs of close embeddings that pushes them away one from the other with a repulsive force proportional to the compression of the springs (you may have a look at [Hooke's law](https://en.wikipedia.org/wiki/Hooke's_law)).

In practice, {{< citet "sablayrolles2018spreading" >}} have illustrated the impact of the KoLeo regularization when paired with another loss function. 
Given a final objective of the form $\mathcal L(f_\theta(\vx), \vy) = \mathcal L_\text{originale}(f_\theta(\vx), \vy) + \lambda \mathcal L_\text{KoLeo}(f_\theta(\vx))$, they showed that the embeddings distribution converges to a uniform distribution with increasing $\lambda$: 

![Figure 1](./koleo_illustration.avif)

You may notice nat this regularization only works for embeddings that belong to a **bounded support**; if not, the regularization can push them away to infinity.

But where does this formula come from?
The KoLeo regularizer is derived from the *Kozachenko–Leonenko differential entropy estimator* ({{< citep "kozachenko1987sample" >}}).

Back in your physics or information classes, you may have seen that the entropy of a discrete distribution is
a measure of its information content, and among all distributions, it is the uniform distribution that achieves maximal entropy. 
Shannon attempt to extend the definition of entropy to continuous variables yielded the **differential entropy**, that is just the regular entropy with the sum swapped for an integral.
Given $X \sim f$, it writes:

$$ h(X) = \E \left[ - \log f(X) \right] = -\int_\sX f(\vx) \log f(\vx)d\vx.$$ 

While differential entropy does not retain all the convenient properties of its original counterpart (it is not always positive, it is not dimensionless), 
it does conserve one that is of interest to us: **for a fixed support $S$, then among all densities on $S$, the uniform distribution maximized the differential entropy.**

The connection with our initial problematic is immediate: to spread embeddings accross the latent space, an option is to **maximize the differential entropy of the embedding distribution**.

And it turns out that the Kozachenko–Leonenko differential entropy estimator - as its name suggests - is precisely an estimator of the differential entropy of a distribution given only samples from it.
Given samples $\vx_0, \dots, \vx_N \in \R^d$ i.i.d from a distribution $\rho$ over a finite support, the Kozachenko–Leonenko estimator writes:

$$ H_N = \frac{d}{N+1} \sum_{i=0}^N {\log \min_{j \neq i} || \vx_i - \vx_j ||} + \log N + \gamma + \log V_1,$$

where $V_1 = \int_{\sB(0, 1)}$ is the volume of the unit-ball and $\gamma$ is [Euler's constant](https://en.wikipedia.org/wiki/Euler's_constant).
Assuming that $N$ (i.e. the batch size) and $d$ (the embedding dimension) are constants in a deep learning framework,
we see that maximizing $H_N$ is equivalent to maximizing $\mathcal{L}_\text{KoLeo}$ (which are, with those considerations, equal up to a negative scale and an offset).

## An intuitive connection between the closest neighbour and the differential entropy
Imagine that you are at your favorite band's biggest show of the decade.
People density in the stadium would probably look something like this:

![Figure 2](./stadium.avif)

The left view shows individual people, while the right figure shows the underlying room density.
Naturally, the latter is non-uniform: everyone wants to be as close as possible to the band, and back corners draw less attention than the front of the stage.

Say you are located at position $\vx_1 \in \R^2$, and the $N-1$ other people enjoying the show with you have respective 2D coordinates $\vx_2, \dots, \vx_N$.
We define the density function of the place $d: \R^2 \longmapsto \R_+$ (right view of the precedent figure)
as the function that gives you the local density at a point $\vx$ in persons per unit of surface.

The intuitive proof of the KoLeo regularization is to realize that your "personal space" can be expressed in two different ways:
- On one hand, it is the reciprocal of the density at your location, i.e $1/d(x_1)$.
For example, if $1/d(\vx_1)=4\text{people}/m^2$, then your personal space is $1/d(\vx_1)$, i.e. $0.25 m^2/\text{people}$.
This is a theoretical approach, that we could compute if we have access to the convenient $d$ function.
- On the other hand, we could also define your personal space using the distance that separates you from your nearest neighbor.
For example, say the closest person to you is $\vx_i$: 
then your individual space is roughly a circle of radius $R=||\vx_1 - \vx_i||_2$, which has an area of $\pi R^2$.
This is an experimental approach, that we have access to using actual data - or people - sampled from the theoretical distributions.

I should add a grain of salt here: defining your personal space using a circle is arbitrary; we could as well use a square or any other shape.
The point is that this surface - however complicated it is - depends on the distance that separates you from the closest music lover.
In a more general framework, we can write that your personal space is $AR^2$, where $A$ is an unknown constant.

As those two views define the same quantity, we have:

$$\frac{1}{d(\vx_1)} \simeq A R_1^2$$

In this illustration, the concept of "people density" is analogous to the concept of probability density function.
They are equivalent, up to the number of samples: $d(\vx) = Np(\vx)$ (probability functions sum to one while densities sum to the total number of people).
We can then write:

$$\frac{1}{Np(\vx_1)} \simeq AR_1^2$$

Composing with a log:

$$-\log N -\log p(\vx_1) \simeq \log(A) + 2\log R_1$$

$$-\log p(\vx_1) \simeq \log(AN) + 2\log R_1$$

We have successfully isolated $-\log p(\vx_1)$, that is the core element of computing the differential entropy $h(p) = \E [ -\log p ]$.

Finally, the same derivations hold true for any other people in the room.
Given that N is large, we can approximate $h(p)$ by its empirical expectation $-1/N\sum_{i=1}^N \log p(\vx_i)$.
By definition, $R_i$ is the distance between $\vx_i$ and its closest neighbour, i.e. $R_i = \min_{j \neq i} || \vx_i - \vx_j ||_2 $.
This yields the final result:

$$h(p) \simeq \frac{-1}{N}\sum_{i=1}^N \log p(\vx_i) \simeq 2 \sum_{i=1}^N \log \min_{j \neq i} || \vx_i - \vx_j ||_2 + \log N + \log A $$

The KoLeo regularization directly derives from this formula;
{{< citet "sablayrolles2018spreading" >}} added a minus sign (so minimizing it maximizes the differential entropy), and removed $A$ and $N$ (the former is a fixed constant, and the latter is fixed by the batch size when training).
We have reached the final formula:

$$\gR_\text{KoLeo} = -\sum_{i=1}^N \log \min_{j \neq i} || \vx_i - \vx_j ||_2 .$$

## The missing constants
While previous derivations establish an intuitive explanation of the final version of the KoLeo regularization,
they strongly rely on the unproven connection between $R$ and $p$.
Let's approach this more rigorously.

Assume we have $N$ points $\vx_i, \dots, \vx_N$ sampled from a distribution with probability distribution function $f$ on a bounded space $\gE$.
Let $i \in \{ 1, \dots, N \}$, and let $R_i = \min_{j \neq i} || \vx_i - \vx_j ||_2 $ be the distance of $\vx_i$ to its nearest neighbor.
Let's analyze the cumulative distribution function of $R_i$:

$$
\begin{split}
\sP (R_i \leq \alpha \mid \vx_i) 
&= 1 - \sP (R_i > \alpha \mid \vx_i) \\
&= 1 - \prod_{j \neq i }\sP (|| \vx_i - \vx_j ||_2 > \alpha \mid \vx_i) \\
&= 1 - \prod_{j \neq i }\left(1 - \sP (|| \vx_i - \vx_j ||_2 \leq \alpha \mid \vx_i) \right) \\
\end{split}
$$

For $\vx_i$ fixed, the probability $\sP (|| \vx_i - \vx_j ||_2 \leq \alpha \mid x_i)$  is the probability of $\vx_j$ being inside the ball of radius $\alpha$ centered around $\vx_i$:

$$\sP (|| \vx_i - \vx_j ||_2 \leq \alpha \mid \vx_i) = \int_{\gB (\vx_i, \alpha)} f(\vx) d\vx$$

where $\gB(\vc, r)$ is the ball of radius $r \geq 0$ and center $\vc \in \R^d$.

If the radius of our ball is small, i.e. $r \ll 1$, the continuity of $f$ implies that it is roughly equal to $f(\vc)$ inside the ball $\gB(\vc, r)$.
Indeed, let $\epsilon > 0$, then the continuity of $f$ implies there exists an $r > 0$ such that $\forall \vx \in \gB(\vc, r), |f(\vx) - f(\vc)| \leq \epsilon$.
Thus, we have:

$$
\begin{split}
\left| \int_{\gB(\vc, r)}{f(\vx)d\vx} - f(\vc)V_r \right|
&= \left| \int_{\gB(\vc, r)}{f(\vx) - f(\vc) d\vx} \right| \\
&\leq \int_{\gB(\vc, r)}{ \left|f(\vx) - f(\vc) \right| d\vx}  \\
&\leq \int_{\gB(\vc, r)}{ \epsilon d\vx} \leq \epsilon V_r
\end{split}
$$

Hence, the value of the integral is approximately $f(\vc) V_r$ where $V_r$ is the volume of $\gB(\vc, r)$.
We also remark that the volume of any ball $V_r$ is equal to the volume of the unit ball $V_1$ scaled by its radius power $d$,
i.e. $V_r = r^d V_1$ (proof by substitution).

Let's assume that $N \gg 1$; it simply means that we have access to many samples.
This is both required - otherwise, the distance to the closest neighbour would not be relevant to the local value of the density -
and acceptable - in the previous concert analogy, there are many people in the room.
We can exploit this assumption to make the radius decrease:
if we switch $R_i$ for $NR_i$ in our derivations, 
we end up with $\sP (NR_i \leq \alpha \mid \vx_i) = \sP (R_i \leq \alpha/N \mid \vx_i)$, 
meaning we have a decreasing $r \to 0$ when $N \to \infty$:

$$\sP (|| \vx_i - \vx_j ||_2 \leq \alpha / N \mid \vx_i) \underset{N \to \infty}{\simeq} f(\vx_i)V_{\alpha / N}
= f(\vx_i)V_1 \left(\frac{\alpha}{N} \right)^d$$

And then:

$$
\begin{split}
\sP (NR_i \leq \alpha \mid \vx_i) 
&\underset{N \to \infty}{\simeq} 1 - \prod_{j \neq i }\left(1 - f(\vx_i)V_1 \left(\frac{\alpha}{N} \right)^d \right) \\
&= 1 - \left(1 - f(\vx_i)V_1 \left(\frac{\alpha}{N} \right)^d \right)^{N-1} \\
\end{split}
$$

This final term may remind you of the proposition $(1 + \lambda /n)^n \underset{n \to \infty}{\rightarrow} e^{\lambda}$ that you may have encountered in your calculus textbook.
It is the last piece of the puzzle.
If we once again switch the analyzed variable $NR_i$ for $(N-1)R_i^d$, we can apply this equality, meaning:

$$
\begin{split}
\sP ((N-1)R_i^d \leq \alpha \mid \vx_i) 
&= \sP (R_i \leq \left( \frac{\alpha}{N-1} \right)^{1/d} \mid \vx_i)  \\
&\underset{N \to \infty}{\simeq} 1 - \left(1 - f(\vx_i)V_1 \frac{\alpha}{N-1} \right)^{N-1} \\
&\underset{N \to \infty}{\simeq} 1 - \exp \left(- f(x_i)V_1 \alpha \right)
\end{split}
$$

We recognize the **cumulative distribution function of an exponential distribution**:
the random variable $(N-1)R_i^d$ converges in probability to an exponential distribution of rate $f(\vx_i)V_1$.

As our objective is to connect $\log R_i$ to $\log f(\vx_i)$, it is natural to consider the logarithm of an exponentially distributed random variable.
If $Z \sim \text{Exp}(\lambda)$, then:

$$\E \left[ \log Z \right] = \log \lambda + \gamma,$$

where $\gamma$ is the Euler constant.
Concluding the proof is a matter of using this relation with our random variable
$(N-1)R_i^d \sim \text{Exp}\left(f(\vx_i)V_1 \right)$:

$$
\begin{aligned}
\E \left[ \log \left( (N-1)R_i^d \right) \mid x_i \right] &= \log (f(\vx_i) V_1) + \gamma \\
\implies -\log f(\vx_i)  &=  -\E \left[ \log R_i^d \mid \vx_i \right] - \log(N-1) + \log V_1 + \gamma
\end{aligned}
$$

We drop the conditioning on $\vx_i$ using the expected value over $\vx_i \sim f$:

$$
-\E \left[ \log f(\vx) \right] =  -\E \left[ \log R^d \right]  - \log(N-1) + \log V_1 + \gamma
$$

And the hypothesis $N \gg 1$ allows us to approximate both expected values by their Monte-Carlo estimators:

$$
\begin{split}
-\frac{1}{N} \sum_{i=1}^N \log f(\vx_i) &= \frac{-1}{N} \sum_{i=1}^N \log R_i^d - \log(N-1) + \log V_1 + \gamma \\
&= \frac{-d}{N} \sum_{i=1}^N \log \min_{j \neq i} || \vx_i - \vx_j ||_2 - \log(N-1) + \log V_1 + \gamma
\end{split}
$$

And we reach the final form of the Kozachenko–Leonenko entropy estimator.
For additional information (and more rigorous demonstrations) about this estimator,
we redirect the reader to {{< citet "delattre2017kozachenko" >}}.

## Implementation
The KoLeo regularization can be implemented in a few lines of Python code with PyTorch.
Note however that it has a $\gO(N^2)$ memory cost due to the pairwise distance computation.

```python
def koleo_reg(x : torch.Tensor, eps:float=1e-9) -> torch.Tensor:
    """Compute the KoLeo regularization, as defined by Sablayrolles et al. (2019)"""
    d = torch.cdist(x, x, p=2).fill_diagonal_(float('inf'))
    r = torch.amin(d, dim=1)
    return -torch.log(r + eps).mean()
  ```