---
title: "The Ko-Leo regularization"
date: 2026-01-15
draft: true
bib:
  - id: "scott1985averaged"
    title: "Averaged Shifted Histograms: Effective Nonparametric Density Estimators in Several Dimensions"
    author: "Scott, David W"
    year: 1985
    journal: "The Annals of Statistics"
    url: "https://www.jstor.org/stable/2241123"
---

Today, I would like to share with you one of my favorite tool 

## A small motivation
The probabilistic framework 
It is a common framework in deep learning to modelize our data as samples of an unknown superior distributions.
Those distributions usualy live in high dimensional world: a small square and grayscale image of size 128 lives in a space of dimension $128\times 128 = 16,384$.
Not even considering that an average modern phone (in 2026, sorry for the futur) saves photo with 12 Megapixels...
The real distribution of "all the possible cat images" for instance, is perfectly imaginary and unatractable...

Luckely, we while we do not know precisely the absolute distribution of cat images, we can grasp a fairly high amount of images of cats, and we can assume those are samples from this high up real distribution.
Now the challenge become: given only samples and not a direct access to $p$, can we still estimate some quantities over $p$ ?

A few ar really easy: take the mean of the distribution for instance.
An unbiased optimal estimator of the mean is simply the empirical mean:
Same for the variance, we have good and optimal estimator to estimate it.
But some can be harder, what about the mode or the median ?
And what about the quantity that interest us today: the entropy of p (or more precisely, the differential entropy, given that samples of p live in a continuous space).

So the first question we will ask today is 

## A quick proof by hand
You have been waiting for this moment for so long and now it is finaly happening:
you are at your favorite artist biggest show of the decade (which must be the Red Hot Chillie Paper, I have no doubt about it).
Here is a plan of the stade de france with the density of the crowd:

Naturally, the room density is non-uniform - everyones want's to be as close as possible to - and back corners draw less attention than the front of the stage.

Say you are located at position $x_1 \in \R^2$, and the $N-1$ other people enjoying the show with you have have respective 2D coordinates $x_2, \dots, x_N$.
Say we have access to the density function of the room $d: \R^2 \longmapsto \R_+$, that for a position $x$ gives you the rough density of people at this position, in number of people per squared meter.

What is you personal space ? 
- One one way, this is the reciprocal of the density at your place: if around you $d(x_1)$ is $4p/m^2$, then your personal space is $1/d(x_1): 0.25 m^2/p$.
- But we could also define this using the distance of your nearest neighbor.
Say the closest person to you is $x_i$: then your individual space is roughly a circle of radius $R=||x_1 - x_i||_2$, which has an area of $\pi R^2$.

To be fair: picking your personnal volume to be a circle is arbitrary: it may has well be a square - with an area of $R^2$ - or any weird shape whose surface can be complicated.
What we can agree on however, is that this surface depends on the distance with the closest musiclover, so in some general sens we can say that your personal surface si $AR^2$ with $A$ an unknown constant.

If we assume those two ways of definin your personal spaces are equal, we have:

$$
\frac{1}{d(x_1)} \simeq A R_1^2
$$

Note that obviously, given their name, the concept of density is closely related to the concept of probability density function.
In fact, they are both the same up to the number of samples: $d(x) = Np(x)$. 
Integrating $p$ over the entire admissible space would give $\int_\text{stade de France} p = 1$, while $\int_\text{stade de France} d = N$.
This yield the following derivation:

$$
\frac{1}{Np(x_1)} \simeq AR_1^2
$$

Using the log:

$$-\log N -\log p(x_1) \simeq \log(A) + 2\log R_1$$

$$-\log p(x_1) \simeq \log(AN) + 2\log R_1$$

We have sucessfuly isolated $-\log p(x_1)$, that is the core element of computing the differential entropy $\H(p) = \E [ -\log p ]$ !
Instead of focusing solely on you, we can also use all the others personn in the room, and approximate this term by it's emperical expectation.
Remember that by definition, $R_1$ was the distance to your closest neighbour, thus in more general therm we have $R_i = \argmin_{j \neq i} || x_i - x_j ||_2 $.
This yiels this final result:

$$\H(p) \simeq \frac{-1}{N}\sum_{i=1}^N \log p(x_i) \simeq 2 \sum_{i=1}^N \log \left( \argmin_{j \neq i} || x_i - x_j ||_2\right) + \log N + \log A $$

The KoLeo regularization directly derives from this; they added a minus sign, as we like to think of regularizxation as term we would like to minimize, but here it corresponds to maximize the entropy to spread things out, and they remove the constants $A$ and $N$ (the latter is usually constant, as it is fixed by the batch size), yielding:

$$\sR_\text{KoLeo} = -\sum_{i=1}^N \log \left( \argmin_{j \neq i} || x_i - x_j ||_2\right).$$

## A bit more formal
pass