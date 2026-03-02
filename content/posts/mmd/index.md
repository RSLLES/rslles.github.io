---
title: "Maximum mean discrepancy"
date: 2026-03-02
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
  - id: "gretton2012kernel"
    title: "A Kernel Two-Sample Test"
    author: "Gretton, Arthur and Borgwardt, Karsten M and Rasch, Malte J and Scholkopf, Bernhard and Smola, Alexander"
    year: 2012
    journal: "Journal of Machine Learning Research 13, 723-773"
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
    journal: "Journal of Machine Learning Research"
    url: "https://www.jmlr.org/papers/volume24/21-1289/21-1289.pdf"
  - id: "biggs2023mmd"
    title: "MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting"
    author: "Biggs, Felix and Schrab, Antonin and Gretton, Arthur"
    year: 2023
    journal: "Advances in Neural Information Processing Systems (NeurIPS)"
    url: "https://proceedings.neurips.cc/paper_files/paper/2023/file/edd00cead3425393baf13004de993017-Paper-Conference.pdf"
---

Statistics are all about studying the world of distributions. In this context, a common objective is to measure how two distributions compare to each other;
a question that can be answered using numerous distance functions --- among which are the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) for instance --- but they generally require direct access to the probability distribution functions (p.d.f).

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
Given the previous samples $\mX$ and $\mY$ respectively from $p$ and $q$, we also define $k: \sX \times \sX \longmapsto \R$ to be a **positive definite kernel**, i.e. a two variables function that is *symmetric and positive semidefinite* ({{< citep "aronszajn1950theory" >}}).
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
When $\mX = \mY$, all three terms cancels out and the MMD is null.
We redirect the curious reader to the excellent course of {{< citet "mairal2022kernel" >}} for additional information about kernel methods for machine learning.

In machine learning, a common usage of MMD is to train **generative models**. 
Indeed, if you have access to a dataset, we can call this dataset $\mX = (\vx_1, \dots, \vx_m)$ and hypothesize that they all come from a higher distribution $p$ that you would like to sample.
Then if we define a new distribution $q_\theta$ using a neural network from which we can easily sample from, we can use the MMD as a loss function to bring $q_\theta$ close to $p$ !
Here is an illustration with a toy example where we train a generator to sample from a simple spiral density:

![Figure 1](./intro.avif)

Code and discussion around this example is available in the last section of this blog.

## The theory


Let's start with a basic characterisation in statistics.
Let $\rvx$ be a random variable and $p$ a distribution on a topological space $\sX$.
The [law of the unconscious statistician](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician) (or [*théorème de transfert*](https://www.bibmath.net/dico/index.php?action=affiche&quoi=.%2Ft%2Ftransfert.html) in French) says:

$$\rvx \sim p \Rightarrow \forall h \in \gC(\sX), \E_{\rvx \sim p}[h(\rvx)] = \int_\sX h(\vx) p(\vx) d\vx,$$

where $\gC(\sX)$ is the space of bounded continuous functions with values in $\mathbb{R}$.
What is really of interest is that the reverse implication of this property holds ({{< citep "dudley2002real" >}}), providing a characterization of probability measures. 
If $\rvx \sim p$ and $\rvy \sim q$, then:
$$p=q \Leftrightarrow  \forall h \in \gC(\sX), \E_{\rvx \sim p}[h(\rvx)] = \mathbb E_{\rvy \sim q}[h(\rvy)].$$

Using this characterization, we can see that if $\rvx$ and $\rvy$ do not follow the same distribution, then there must exist a function $h$ such that the expected values of $h(\rvx)$ and $h(\rvy)$ are different.
Naturally, we expect that the more different $\rvx$ and $\rvy$, the easier it should be to find a function $h$ that exacerbates their difference.

The fundamental idea of {{< citet "gretton2012kernel" >}} relies on this insight: they define **the MMD as the greatest possible difference between the expected values of $h(\rvx)$ and $h(\rvy)$ for a function $h$ in $\gH$ that maximizes this precise quantity**. In mathematical terms:

$$\operatorname{MMD}[\gH, p, q] = \sup_{h \in \mathcal \gH} \left( \E_{\rvx \sim p}[h(\rvx)] - \E_{\rvy \sim q}[h(\rvy)]  \right).$$

Of course, finding a supremum over a space of functions is far from trivial.
But this apparently complex optimization problem has a closed form for a very specific class of functions: when $h$ is restricted to the unit ball of a reproducing kernel Hilbert space (RKHS) ({{< citep "aronszajn1950theory" >}}) $\gH$.

Let's pick a RKHS $\gH$.
By the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem),
we denote $\phi: \sX \longmapsto \gH$ the feature mapping of $\gH$ and the kernel $k(\vx, \vy) = \dotprod{\phi(\vx)}{\phi(\vy)}_\gH$ for $\vx, \vy \in \sX^2$.

Let $h$ be a function of $\gH$.
For all $\vx \in \sX$, we have $h(\vx) = \dotprod{\phi(\vx)}{h}_\gH$.
We can use this property to derive a formula for $\E_{\rvx \sim p}[h(\rvx)]$:

$$
\begin{split}
\E_{\rvx \sim p} (h(\rvx)) &= \int_{\vx \in \sX}{h(\vx)p(\vx)dx} = \int_{\vx \in \sX}{\dotprod{\phi(\vx)}{h}_\gH p(\vx)dx} \\
&=  \dotprod{\int_{\vx \in \sX} \phi(\vx) p(\vx)dx}{h}_\gH = \dotprod{\mu_p}{h}_\gH,
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
to prove some theory, but in the end to get back to simple expressions directly involving evaluation of the kernel $k: \sX \times \sX \longmapsto \R$, which is tractable.

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

## Linear MMD

The previous derived expression is not suited for online usage.
Indeed, for any new point added to the dataset, we need to compute its application through the kernel against all previous points,
meaning we need to keep the entire history, and have a quadratic scaling with respect to the number of samples.

If we look at the previous formula, we see that **the minimum number of samples required by the MMD is 2 from each distribution**.
Given only $\vx_1, \vx_2$ from $p$ and $\vy_1, \vy_2$ from $q$, we can approximate
$\E_{\rvx_1 \sim p, \rvx_2 \sim p}[k(\rvx_1, \rvx_2)]$ by $k(\vx_1, \vx_2)$,
$\E_{\rvy_1 \sim q, \rvy_2 \sim q}[k(\rvy_1, \rvy_2)]$ by $k(\vy_1, \vy_2)$,
and $\E_{\rvx \sim p, \rvy \sim q}[k(\rvx, \rvy)]$ by $\frac{1}{2}(k(\vx_1, \vy_1) + k(\vx_2, \vy_2))$.
This yields the lightest possible estimator for the squared MMD, 
that we denote as $\hat{\operatorname{MMD}}^2_2$ and is:

$$
\hat{\operatorname{MMD}}^2_2[\vx_1, \vx_2, \vy_1, \vy_2] = k(\vx_1, \vx_2) - k(\vx_1, \vy_1) - k(\vx_2, \vy_2) + k(\vy_1, \vy_2).
$$

This estimate is of course highly noisy --- being based on two samples --- 
but is also extremely cheap to compute.
**Linear MMD leverages this formulation by computing online $\hat{\operatorname{MMD}}^2_2$ for each new pair of points, and averages their value over time to create a linear estimator.**
Assuming we have $2n$ samples $(\vx_1, ..., \vx_{2n})$ from $p$ and also $2n$ samples $(\vy_1, ..., \vy_{2n})$ from $q$, we can define the linear MMD estimator as:

$$
\begin{split}
\hat{\operatorname{MMD}}_\text{linear}^2[\mX, \mY] 
&= \frac{1}{n}\sum_{i=1}^n \hat{\operatorname{MMD}}^2_2[\vx_{2i}, \vx_{2i+1}, \vy_{2i}, \vy_{2i+1}] \\
&= \frac{1}{n}\sum_{i=1}^n \left[ k(\vx_{2i}, \vx_{2i+1}) + k(\vy_{2i}, \vy_{2i+1}) - k(\vx_{2i}, \vy_{2i+1}) - k(\vx_{2i+1}, \vy_{2i}) \right].
\end{split}
$$

This formulation enables online MMD estimation: it requires $\gO(1)$ memory cost and $\gO(n)$ computational cost, to the detriment of higher variance.

With large datasets in machine learning, we may also encounter scenarios where the total number of samples is too high to compute the quadratic classic MMD, but a higher-order MMD than the simple $\hat{\operatorname{MMD}}_2$ can be computed.
You can generalize the previous idea by fixing a block size $b$ specific to your compute budget, and use the following estimator:

$$
\hat{\operatorname{MMD}}_{b\text{-linear}}^2[\mX, \mY] 
= \frac{1}{n}\sum_{i=1}^n \hat{\operatorname{MMD}}^2_{b}[(\vx_{ib+1}, \dots, \vx_{(i+1)b}), (\vy_{ib+1}, \dots, \vy_{(i+1)b})] \\
$$

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
Therefore, the next section is about the latest badnwitch selection methods I could find at the time of writting to pick great kernerls.

## Kernel selection methods
The performance of the MMD varies highly depending on the choice of the kernel.
While the Gaussian kernel --- also known as radial basis function kernel (RBF) --- is almost ubiquotous with the Laplace kernel, selecting the correct sigma --- also known as the bandwidth --- is critical.

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

You may notive that this code re-computes the percentiles for each new batch.
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
aligning kernel selection with the MMD philosophie.
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
