---
title: "The REINFORCE algorithm"
date: 2026-05-24
draft: false
bib:
  - id: "williams1992simple"
    title: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
    author: "Williams, Ronald J"
    year: 1992
    journal: "Machine Learning"
    url: "https://link.springer.com/article/10.1007/BF00992696"
  - id: "bengio2013estimating"
    title: "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"
    author: "Bengio, Yoshua and Léonard, Nicholas and Courville, Aaron"
    year: 2013
    journal: "arXiv preprint 1308.3432"
    url: "https://arxiv.org/abs/1308.3432"
  - id: "kingma2013auto"
    title: "Auto-Encoding Variational Bayes"
    author: "Kingma, Diederik P and Welling, Max"
    year: 2014
    journal: "International Conference on Learning Representations (ICLR)"
    url: "https://arxiv.org/abs/1312.6114"
  - id: "maddison2016concrete"
    title: "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"
    author: "Maddison, Chris J and Mnih, Andriy and Teh, Yee Whye"
    year: 2017
    journal: "International Conference on Learning Representations (ICLR)"
    url: "https://arxiv.org/abs/1611.00712"
  - id: "jang2016categorical"
    title: "Categorical Reparameterization with Gumbel-Softmax"
    author: "Jang, Eric and Gu, Shixiang and Poole, Ben"
    year: 2017
    journal: "International Conference on Learning Representations (ICLR)"
    url: "https://arxiv.org/abs/1611.01144"
  - id: "schulman2017proximal"
    title: "Proximal Policy Optimization Algorithms"
    author: "Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg"
    year: 2017
    journal: "arXiv preprint"
    url: "https://arxiv.org/abs/1707.06347"
  - id: "kool2019buy"
    title: "Buy 4 REINFORCE Samples, Get a Baseline for Free!"
    author: "Kool, Wouter and van Hoof, Herke and Welling, Max"
    year: 2019
    journal: "International Conference on Learning Representations (ICLR)"
    url: "https://openreview.net/forum?id=r1lgTGL5DE"
  - id: "shao2024deepseekmath"
    title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
    author: "Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, YK and Wu, Yang and others"
    year: 2024
    journal: "arXiv preprint"
    url: "https://arxiv.org/abs/2402.03300"
---

Nowadays, most supervised deep learning frameworks rely on first-order optimization methods, which require the loss function to be differentiable.
Unfortunately, a large class of problems leads to non-differentiable loss functions (a common blocking behavior is sampling, for example.)
Think about the problem of policy optimization for example: you search for an optimal strategy over a set of discrete actions.
Here, something as simple as a binary decision --- that can be modeled by a Bernoulli trial --- is not differentiable and breaks the default deep learning framework.

In today's post, I would like to tackle a mathematical identity called REINFORCE ({{< citep "williams1992simple" >}}) that precisely **allows to optimize what you can't differentiate by leveraging a stochastic process**.
It is the secret engine behind numerous modern tools --- PPO ({{< citep "schulman2017proximal" >}}) or GRPO ({{< citep "shao2024deepseekmath" >}}) for example --- and I think it's an important tool to install in your machine learning toolkit.

## The core formulation

To formally introduce REINFORCE, let's consider a problem of finding the optimal parametric distribution $p_\theta$ such that its samples minimize in expectation a loss function $\Ls$.

$$ \theta^* = \argmin_{\theta \in \Theta} F(\theta) \; \text{where} \; F(\theta) = \E_{\rvx \sim p_\theta} \left[ \Ls(\rvx) \right] .$$


This formulation connects with the previous motivations: 
we are looking for a parametric distribution $p_\theta$ --- often called *policy* in the context of optimal control --- that produces samples $\rvx \sim p_\theta$ --- often called *trajectories* ---  which yield in expectation the best performance according to some loss function $\Ls$.

Alternatively, when $\Ls(\rvx)$ is *not* differentiable, we introduce a surrogate distribution $p_\theta$ that serves as an exploration tool.

To optimize this objective with first-order optimization methods --- think of gradient descent ---, we need access to $\nabla_\theta F(\theta)$. Naively computing this quantity gives:

$$ \nabla_\theta F(\theta) = \nabla_\theta \int_{X} \Ls(x) p_\theta(x) dx = \int_{X} \Ls(x) \nabla_\theta p_\theta(x) dx \label{naive}.$$

Swapping the gradient with the integral comes from the [Dominated convergence theorem](https://en.wikipedia.org/wiki/Dominated_convergence_theorem) (DCT).
This integral may be tractable when $p_\theta$ is a very simple distribution, like a Gaussian, but this is unsuited for larger real-world scenarios when distributions can be much more complex, like trajectories or policies for games.

The key identity that unlocks the path to the REINFORCE estimator is the [log-derivative trick](https://en.wikipedia.org/wiki/Logarithmic_derivative).
For a function $f$, a direct application of the chain rule gives $\nabla \log f = \frac{\nabla f}{f}$, meaning we have $\nabla f = f \nabla \log f$.
If we apply this formula in {{< eqref "naive" >}}:

$$ 
\begin{split}
\nabla_\theta F(\theta) &= \int_{X} \Ls(x) \nabla_\theta p_\theta(x) dx = \int_{X} \Ls(x) p_\theta(x) \nabla_\theta \log p_\theta(x) dx \\ 
&= \int_{X} \left [ \Ls(x) \nabla_\theta \log p_\theta(x) \right] p_\theta(x) dx =  \E_{\rvx \sim p_\theta} \left[ \Ls(\rvx) \nabla_\theta \log p_\theta(\rvx) \right].
\end{split}
$$

As you can see, we have rewritten the gradient of $F(\theta)$ as the expectation of a new random variable 
$\rvz = \Ls(\rvx) \nabla_\theta \log p_\theta(\rvx)$,
and naturally $\E[\rvz]$ can be approximated by a Monte-Carlo estimator.

## Improving the variance

Now, while this algorithm is unbiased, meaning it does converge to the true gradient, it can have a gigantic --- sometimes infinite --- variance, meaning it is extremely noisy.
And indeed, looking at the REINFORCE gradient estimate, the gradient signal is proportional to the value of $\Ls(\rvx)$.
So what if we could modify $\Ls(\rvx)$ such that the bad values are negative, meaning they discourage, and the good ones are positive, meaning they push in the good direction?

This turns out to be a good idea and is termed a **baseline**: instead of minimizing $\Ls$, we optimize $\Ls - b$, where $b$ is a carefully chosen number that approximates the mean value of $\Ls$. 
Mathematically, we can prove that adding such a baseline changes nothing to the problem.
Let's decompose it:

$$ \nabla_\theta \E_{\rvx \sim p_\theta} \left[ \Ls(\rvx) - b\right] = \nabla_\theta \E_{\rvx \sim p_\theta} \left[ \Ls(\rvx)\right] - \nabla_\theta\E_{\rvx \sim p_\theta} \left[ b \right]. $$

Nothing fancy here, the expectation and the gradient are linear. 
But if you consider the second term, with the objective being a constant $b$, then it is the gradient of a stochastic objective ... where the objective does not depend on the sampled value? So naturally, the expectation should be a constant here, equal to $b$, and then the gradient would be 0.
We can prove this in two lines:

$$ \nabla_\theta\E_{\rvx \sim p_\theta} \left[ b \right] = \nabla_\theta \int_X b \, p_\theta(x)dx = b \nabla_\theta \underbrace{\int_X p_\theta(x) dx}_{=1} = b \nabla_\theta 1 = 0.$$

So adding an offset to $\Ls$ does not change the optimization of the objective, just like adding a constant to a differentiable objective does not change the gradient; it is the same underlying idea.

A good first step would be to pick the baseline $b$ that minimizes the variance of $\rvz(b) = (\Ls(\rvx) - b) \nabla_\theta \log p_\theta(\rvx)$.
We have:

$$\Var(b)
= \E\left[ \|\rvz(b)\|_2^2 \right] - \|\E[\rvz(b)]\|_2^2 
= \E\left[ (\Ls(\rvx) - b)^2 \|\nabla_\theta \log p_\theta(\rvx)\|_2^2 \right] + \cst.$$

Indeed, the baseline does not change the expected value, i.e. $\E[ \rvz(b)] = \E[ \rvz(0)]$,
thus the second term is independent of $b$.
Taking the derivative with respect to $b$ and setting it to zero:

$$\frac{d \Var}{db} = -2\E\left[(\Ls(\rvx) - b) \|\nabla_\theta \log p_\theta(\rvx)\|_2^2 \right] = 0.$$

Solving for $b$:

$$b^* = \frac{\E\left[\Ls(\rvx) \|\nabla_\theta \log p_\theta(\rvx)\|_2^2 \right]}{\E\left[\|\nabla_\theta \log p_\theta(\rvx)\|_2^2\right]}.$$

The optimal baseline is the weighted expectation of $\Ls(\rvx)$ with weights $\|\nabla_\theta \log p_\theta(\rvx)\|_2^2 / \E\left[\|\nabla_\theta \log p_\theta(\rvx)\|_2^2\right]$. 
These weights are hard to estimate in practice, so a good first choice for a baseline is simply the expected reward itself, $\E[\Ls(\rvx)]$.

### One evaluation? Use an EMA baseline

As we have seen so far, a good strategy is to target $\E_{\rvx \sim p_\theta}[\Ls(\rvx)]$.
But unfortunately, this quantity is moving as we optimize the objective, which shifts the estimate of $\theta$.
A natural idea is to use an Exponential Moving Average (EMA) to track a moving estimate of this quantity.
Hence, we could implement reinforce with this trick:

```python
class BaselineREINFORCE(nn.Module):
    """REINFORCE loss function with the ema-baseline variance stabilizing trick."""

    def __init__(
        self,
        lambd: float = 0.01,
        eps: float = 1e-9,
        ensure_detached_loss: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.ensure_detached_loss = ensure_detached_loss
        self.ema_mean = ExpMovingAverage(lambd=lambd, n_elements=1)
        self.ema_var = ExpMovingAverage(lambd=lambd, n_elements=1)

    def forward(self, loss: Tensor, log_prob: Tensor) -> Tensor:  # noqa: D102
        if self.ensure_detached_loss:
            loss = loss.detach()

        reward = loss.mean()

        if self.ema_mean.is_initialized:
            mean = self.ema_mean.value()
            std = self.ema_var.value().sqrt()
            advantage = (loss - mean) / (std + self.eps)
        else:
            advantage = loss - loss.mean()

        self.ema_mean.update(reward)
        residual_sq = (reward - self.ema_mean.value()).pow(2)
        self.ema_var.update(residual_sq)

        return (advantage * log_prob).mean()
```

### More than one? Use a leave-one-out baseline

Machine learning often processes multiple values within a mini-batch, hence it is not uncommon to have access to multiple loss evaluations.
So say you have $N$ loss evaluations $\Ls(\vx_i)$.
Then, a natural unbiased baseline for the i-th evaluation is 
$$b_i = \frac{1}{N-1}\sum_{j=1, j \neq i}^N \Ls(\vx_j) = \frac{1}{N-1}\left(  \left[ \sum_{j=1}^N \Ls(\vx_j) \right] - \Ls(\vx_i) \right),$$

which means the loss function becomes:

$$\Ls(\vx_i) - b_i = \frac{1}{N-1}\left(  N \Ls(\vx_i) - \sum_{j=1}^N \Ls(\vx_j) \right). $$
The fact that we drop only the current element from the mean estimation --- hence making it unbiased --- is called the leave one out trick, and was first introduced by {{< citet "kool2019buy" >}}.
Additionally, with more samples, one may normalize by the group standard deviation

$$\sigma_i = \sqrt{\frac{1}{N-2} \sum_{j=1, j \neq i}^N \left( \Ls(\vx_j) - b_i \right)^2},$$

so the quantity $z_i = (\Ls(\vx_i) - b_i) / \sigma_i$ is approximately normally distributed, which keeps updates at a stable scale.

Here is a potential implementation: 

```python
class LOOReinforce(nn.Module):
    """REINFORCE loss function with the leave-one-out variance stabilizing trick [2]."""

    def __init__(
        self,
        ensure_detached_loss: bool = True,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.ensure_detached_loss = ensure_detached_loss
        self.eps = eps

    def forward(self, losses: Tensor, log_prob: Tensor):  # noqa: D102
        if losses.ndim != 1:
            raise ValueError(f"Expect 1D tensor for losses, got {losses.shape}")
        if losses.size(0) < 3:
            raise ValueError(f"Expect at least 3 losses, got {losses.size(0)}")
        N = losses.size(0)

        if self.ensure_detached_loss:
            losses = losses.detach()

        rolls = losses.repeat(2).as_strided((N, N), (1, 1))
        loo = rolls[:, 1:]
        std, mean = torch.std_mean(loo, dim=1)
        losses = (losses - mean) / (std + self.eps) * log_prob
        return losses.mean()
```

## Modern usage of REINFORCE

### PPO

Proximal Policy Optimization (PPO, {{< citet "schulman2017proximal" >}}) is the algorithm behind
Reinforcement Learning from Human Feedback (RLHF), the methodology used to fine-tune ChatGPT and
other instruction-following LLMs.

In this setting, the **policy** is the LLM generating sequences of tokens, and the **reward**
comes from a reward model trained on human preference comparisons. Since text generation involves
discrete token sampling --- a non-differentiable operation --- the gradient of the reward with respect
to the policy parameters cannot be computed directly. This is precisely where REINFORCE is needed.

PPO is a REINFORCE method with two key contributions:

1. **Clipping the gradient.** In vanilla REINFORCE, large updates occur when an unlikely sample
   has a high reward: the log-probability term $\nabla_\theta \log p_\theta(\vx)$ is large for
   low-probability samples, and multiplied by a large reward $\Ls(\vx)$, this causes instability.
   PPO avoids this by clipping the probability ratio $r = p_\theta(\vx) / p_{\theta_\text{old}}(\vx)$ to stay within $[1-\varepsilon, 1+\varepsilon]$, preventing any single update from moving too far from the previous policy.

2. **Critic as a learned baseline.** Rather than a global constant like an EMA, PPO uses a
   *state-dependent* value function $V(s)$, called the **critic** (another LLM). It is typically initialized from the same pretrained LLM, and is trained to predict expected future reward from the current state. The REINFORCE update then uses $V(s)$ as the baseline, which reduces the variance.

Main ideas: clip updates for stability, and use a learned value function as a baseline.
The cost is training and running three networks simultaneously: the policy, the critic, and the
reward model.


### GRPO

Given the computational cost of PPO, DeepSeek came up with a method called Group Relative Policy Optimization (GRPO, {{< citet "shao2024deepseekmath" >}}) that improves on PPO by removing the critic.
They use the same intuition as in our Leave-One-Out part:
instead of generating one completion per prompt, they generate a group of $G$ completions and use their mean reward as the baseline, which is much cheaper than a learned value function.

Note that unlike the [leave-one-out estimator](#more-than-one-use-a-leave-one-out-baseline), the baseline includes the current sample, making it slightly biased, but realistically, the bias is $O(1/G)$ and hence is negligible at typical group sizes ($G \simeq 16$).


## Alternative methods to sample differentiably 

To conclude on a lighter note, here are three other tricks to propagate gradients through a sampling operation.

### The reparametrization trick for continuous random variables

The term "reparametrization trick" was largely popularized by the VAE paper ({{< citep "kingma2013auto" >}}).
The idea is to reformulate a parametric distribution $\rvx \sim p_\theta$ as a deterministic function of an external, non-parametric random variable: $\rvx = g_\theta(\rvz)$ where $\rvz \sim p$.
Since $g_\theta$ is deterministic, you can now backpropagate through it normally.

For instance, to sample from $\gN(\mu, \sigma^2)$ and differentiate with respect to $\mu$ and $\sigma$:
$$\rvx = \mu + \sigma\, \rvz, \quad \rvz \sim \gN(0, 1).$$

More generally, for any 1D distribution, this connects to [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling): pick $\rvz \sim \gU(0, 1)$ and set $\rvx = g_\theta(\rvz) = F_\theta^{-1}(\rvz)$, where $F_\theta$ is the CDF of $p_\theta$.

This works well for continuous distributions, but cannot be applied when sampling from a discrete one.

### Relaxations for discrete RVs

When dealing with a discrete distribution, one option is to relax the discrete constraint with a continuous analog, and then apply the reparametrization trick on that surrogate. Examples include:
- A **Poisson** variable can be approximated by a **Gamma** distribution, which admits a standard reparametrization.
- A **categorical** variable can be approximated by the **Concrete distribution**, also known as the **Gumbel-Softmax** (concurrent works of {{< citet "maddison2016concrete" >}} and {{< citep "jang2016categorical" >}}): replace the argmax over logits with a softmax at temperature $\tau$, injecting Gumbel noise to reparametrize the sampling. As $\tau \to 0$, samples approach one-hot vectors; at larger $\tau$ they are smooth and fully differentiable.

### The straight-through estimator for discrete RVs

A simpler alternative is the **straight-through estimator** ({{< citep "bengio2013estimating" >}}): use the hard discrete value in the forward pass, but pass the gradient straight through the rounding operation as if it were the identity.

Example for a Bernoulli distribution:

```python
mask = torch.bernoulli(p) # p.requires_grad = True but mask.requires_grad = False
mask = mask + p - p.detach() # mask.requires_grad = True with the gradient of p
```

It can be combined with various [relaxations](#relaxations-for-discrete-rvs);
for example, a hard categorical sampling can be paired with a Concrete relaxation:
```python
noise = torch.distributions.Gumbel(0, 1).sample(logits.shape)
soft = torch.softmax((logits + noise) / tau, dim=-1) # concrete relaxation
idx  = soft.argmax(dim=-1, keepdim=True) # hard version 
hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)  # one-hot
y    = hard + soft - soft.detach()  # hard in forward, soft gradient in backward
```

This yields a biased estimator, but it is extremely cheap and often works surprisingly well in practice --- the intuition being that $\rvx$ and $\hat{\rvx}$ are close, so their gradients are approximately equal.
