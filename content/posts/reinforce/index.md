---
title: "The REINFORCE algorithm"
date: 2026-05-09
draft: true
bib:
  - id: "williams1992simple"
    title: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
    author: "Williams, Ronald J"
    year: 1992
    journal: "Machine Learning"
    url: "https://link.springer.com/article/10.1007/BF00992696"
  - id: "schulman2017proximal"
    title: "Proximal Policy Optimization Algorithm"
    author: "Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg"
    year: 2017
    journal: "arXiv preprint"
    url: "https://arxiv.org/abs/1707.06347"
  - id: "shao2024deepseekmath"
    title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
    author: "Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, YK and Wu, Yang and others"
    year: 2024
    journal: "arXiv preprint"
    url: "https://arxiv.org/abs/2402.03300"
---

Nowadays, most supervised deep learning frameworks relie on first order optimization methods, which require the loss function to be differentiable.
But this is not always the case. 
While most functions are differentiable almost everywhere, a common blocking behavior is **sampling**.
Think of policy optimization for example, where you search an optimal strategy over a set of discrete action. If you simulate a discrete action, something as simple as a binary decision --- that can't be modeled by a bernoulli trial --- is not differentiable and breaks your framework.

Hence in today's post, I would like to tackle a mathematical indentity called REINFORCE ({{< citep "williams1992simple" >}}) that precisely allows to optimize what you can't differentiate by leveraging a sampling process.
It is the secret engine behind numerous modern tools --- PPO ({{< citep "schulman2017proximal" >}}) or GRPO ({{< citep "shao2024deepseekmath" >}}) for example --- and I think it should be in the toolkit of every machine learning researcher.

## An intuitive explanation
So a way of optimizing could be by *random* search: we sample some values from a prior distribution, we evaluate how good they are, and then we compare  

## The maths

To formally introduce REINFORCE, let's consider a problem of finding the optimal parametric distribution $p_\theta$ such that it samples minimize in expectation a loss function $\Ls$.

$$ \theta^* = \argmin_{\theta \in \Theta} F(\theta) \; \text{where} \; F(\theta) = \E_{\rvx \sim p_\theta} \left[ \Ls(\rvx) \right] .$$


This formulation connects with the previous motivations: 
we are looking from a parametric distribution $p_\theta$ --- often called *policy* in the context of optimal control --- that produces samples $\vx \sim p_\theta$ --- often called *trajectories* ---  which yield in expectation the best performance according to some loss function $\Ls$.

Alternatively, if you imagine that $\Ls(\rvx)$ is *not* differentiable: hence, we introduce a surroguate distribution $p_\theta$ that serves as an exploration tool.

To optimize this method with 1st order optimization method --- think of gradient descent ---, we need access to $\nabla_\theta F(\theta)$.Naively computing this quantity produces this:

$$ \nabla_\theta F(\theta) = \nabla_\theta \int_{X} \Ls(x) p_\theta(x) dx = \int_{X} \Ls(x) \nabla_\theta p_\theta(x) dx \label{naive}.$$

Swapping the gradient with the integral comes from the [Dominated convergence theorem](https://en.wikipedia.org/wiki/Dominated_convergence_theorem) (DCT).
This integral may tractable when $p_\theta$ is a very simple distribution, like a gaussian, but this is unsuited for bigger real world scenarios when distributions can be much more complex, like trajectories or policies for games ...

For some distributions, a common idea is the reparemtrization trick : assume we can sample from another distribution and turn z into the target distribution, meaning if $z \sim q$ and then for $x = g_\theta (z)$, then $x \sim p_\theta$, it means we can reparametrized.

Another idea is to leverage the [log-derivative trick](https://en.wikipedia.org/wiki/Logarithmic_derivative).
For a function f, a direct application of the chain rules yields $\nabla \log f = \frac{\nabla f}{f}$, meaning we have $\nabla f = f \nabla \log f$.
If we apply this formula in {{< eqref "naive" >}}:

$$ 
\begin{split}
\nabla_\theta F(\theta) &= \int_{X} \Ls(x) \nabla_\theta p_\theta(x) dx = \int_{X} \Ls(x) p_\theta(x) \nabla_\theta \log p_\theta(x) dx \\ 
&= \int_{X} \left [ \Ls(x) \nabla_\theta \log p_\theta(x) \right] p_\theta(x) dx =  \E_{x \sim p_\theta(x)} \left[ \Ls(x) \nabla_\theta \log p_\theta(x) \right]
\end{split}
$$

As you can see, we have rewritten the gradient of $F(\theta)$ as the expectation of a new random variable 
$\rvz = \Ls(\rvx) \nabla_\theta \log p_\theta(\rvx)$,
and naturally $\E(\rvz)$ can be approximated by a Monte-Carlo estimator.

## Improving the variance

Now, while this algorithm is unbiased, meaning it does converge to the true gradient, it can have a gigantic - sometimes infinite - variance, meaning it is extremely extremely noisy.
And indeed, when you look at the previous table:

The force is proportionnal to the value of $\Ls(x)$.
So what if we could modify $\Ls(x)$ such that the bad values are negative, meaning they push forward, and the good ones are positives, meaning they push in the good direction ?

Turns out this is a pretty good idea; and it is termed **baseline**, if instead of minimizing $\Ls$, we optimize $\Ls - b$, where b is a carefully choosen number that tries to approximate the median value of $\Ls$, in a sens. 
Mathematically, we can proove that adding such a baseline changes nothing to the problem.
Let's decompose it:

$$ \nabla_\theta \E_{x \sim p_\theta(x)} \left[ \Ls(x) - b\right] = \nabla_\theta \E_{x \sim p_\theta(x)} \left[ \Ls(x)\right] - \nabla_\theta\E_{x \sim p_\theta(x)} \left[ b \right] $$

Nothing fancy here, the expectation and the gradient are linear. 
But if you consider the second term, with the objective being a constant $b$, then it is the gradient of an stochastic objective ... where the objective does not depend on sampled value ? So naturally, the expectation should be a constant here, equal to b, and then the gradient would be 0.
We can prove this in two lines:

$$ \nabla_\theta\E_{x \sim p_\theta(x)} \left[ b \right] = \nabla_\theta \int_X b p_\theta(x)dx = b \nabla_\theta \underbrace{\int_X p_\theta(x) dx}_{=1} = b \nabla_\theta 1 = 0$$

So adding an offset to $\Ls$ does not change the optimization of the objective, just like adding a constant to an differentiable objective does not change the gradient.

If we have a look at the gradient of this estimator with the baseline:

$$ \sV \left[ \rvz \right] =  \E \left[ ||\rvz||_2^2 \right] - || \E \left[ \rvz \right] ||_2^2 $$


## Alternatives

Alternative optimization methods include:

- The reparametrization trick.
- The pass-through trick.