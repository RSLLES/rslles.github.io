---
title: "Self-supervised learning is about informative signals"
date: 2026-07-01
draft: true
bib:
  - id: "kepler1611"
    title: "Strena seu de Nive Sexangula"
    author: "Kepler, Johannes"
    year: 1611
    journal: "Apud Godefridum Tampach, Frankfurt am Main"
    url: "https://www.thelatinlibrary.com/kepler/strena.html"
  - id: "raissi2019pinn"
    title: "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"
    author: "Raissi, M. and Perdikaris, P. and Karniadakis, G.E."
    year: 2019
    journal: "Journal of Computational Physics"
    url: "https://doi.org/10.1016/j.jcp.2018.10.045"
  - id: "chen2020simple"
    title: "A Simple Framework for Contrastive Learning of Visual Representations"
    author: "Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey"
    year: 2020
    journal: "International Conference on Machine Learning (ICML)"
    url: "https://simclr.github.io/"
  - id: "purushwalkam2020demystifying"
    title: "Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases"
    author: "Purushwalkam, Senthil and Gupta, Abhinav"
    year: 2020
    journal: "Advances in Neural Information Processing Systems (NeurIPS)"
    url: "https://arxiv.org/abs/2007.13916"
  - id: "grill2020bootstrap"
    title: "Bootstrap your own latent: A new approach to self-supervised Learning"
    author: "Grill, Jean-Bastien and Strub, Florian and Altché, Florent and Tallec, Corentin and Richemond, Pierre and Buchatskaya, Elena and Doersch, Carl and Avila Pires, Bernardo and Guo, Zhaohan and Gheshlaghi Azar, Mohammad and others"
    journal: "Advances in Neural Information Processing Systems (NeurIPS)"
    year: 2020
    url: "https://arxiv.org/abs/2006.07733"
  - id: "radford2021learning"
    title: "Learning Transferable Visual Models From Natural Language Supervision"
    author: "Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others"
    journal: "International Conference on Machine Learning (ICML)"
    year: 2021
    url: "https://arxiv.org/abs/2103.00020"
  - id: "robinson2021contrastive"
    title: "Contrastive Learning with Hard Negative Samples"
    author: "Robinson, Joshua and Chuang, Ching-Yao and Sra, Suvrit and Jegelka, Stefanie"
    journal: "International Conference on Learning Representations (ICLR)"
    year: 2021
    url: "https://arxiv.org/abs/2010.04592"
  - id: "chen2021simsiam"
    title: "Exploring Simple Siamese Representation Learning"
    author: "Chen, Xinlei and He, Kaiming"
    year: 2021
    journal: "Conference on Computer Vision and Pattern Recognition (CVPR)"
    url: "https://arxiv.org/abs/2011.10566"
  - id: "caron2021emerging"
    title: "Emerging Properties in Self-Supervised Vision Transformers"
    author: "Caron, Mathilde and Touvron, Hugo and Misra, Ishan and Jégou, Hervé and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand"
    journal: "International Conference on Computer Vision (ICCV)"
    year: 2021
    url: "https://arxiv.org/abs/2104.14294"
  - id: "zbontar2021barlow"
    title: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
    author: "Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, Stéphane"
    journal: "International Conference on Machine Learning (ICML)"
    year: 2021
    url: "https://proceedings.mlr.press/v139/zbontar21a"
  - id: "bardes2022vicreg"
    title: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
    author: "Bardes, Adrien and Ponce, Jean and LeCun, Yann"
    journal: "International Conference on Learning Representations (ICLR)"
    year: 2022
    url: "https://arxiv.org/abs/2105.04906"
  - id: "oquab2023dinov2"
    title: "DINOv2: Learning Robust Visual Features without Supervision"
    author: "Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others"
    journal: "Transactions on Machine Learning Research (TMLR)"
    year: 2024
    url: "https://arxiv.org/abs/2304.07193"
  - id: "balestriero2025lejepa"
    title: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
    author: "Balestriero, Randall and LeCun, Yann"
    journal: "arXiv preprint"
    year: 2025
    url: "https://arxiv.org/abs/2511.08544"
  - id: "mcdonnell2026mutts"
    title: "Mutts"
    author: "McDonnell, Patrick"
    year: 2026
    journal: "King Features Syndicate"
    url: "https://mutts.com/products/strip-012621?variant=41141367603357"
  - id: "oed-snowflake"
    title: "snowflake (n.), sense 1"
    author: "Oxford University Press"
    year: 2026
    journal: "Oxford English Dictionary"
    url: "https://doi.org/10.1093/OED/9647378184"
---

A large part of deep learning consists of training highly parameterized neural networks to map input data to quantities of interest.
For instance, image classification assigns class probability to images, inverse problems estimate initial quantities from corrupted measurements, representation learning attaches relevant embeddings to objects, etc ...
The standard framework in the field involves 1) pick a good class of parametric functions -- i.e. efficient model architectures --, 2) design an objective, a.k.a. a loss function, and 3) minimize it with your favorite optimization algorithm.

This blogpost develops my insights about step 2), especially in the self-supervised setup.
Indeed, while loss functions for supervised training are generally straightforward, 
designing objectives for self-supervised learning that maximize training efficiency remains a complicated task. 
Hence, in this blogpost, I would like to explore **how can we design good learning signals for a given task in a self-supervised setting?**

TL;DR, my intuition is **that loss functions for self-supervised learning come from necessary and, hopefully, sufficient conditions the answer must satisfy**, rather than directly from the clean answer itself as in supervised learning. 
Interestingly, in [logic reasoning]((https://en.wikipedia.org/wiki/Logic)), definitions by conditions vs the complete list of answers are respectfully refered to as intensional vs extensional definitions.
This post starts by unpacking that analogy --- intension vs extension --- then deciphers several self-supervised approaches through this lens, and concludes with practical advice and recipes to apply them in our own projects.

{{< figure src="./intro.gif" attr="*Mutts* by Patrick McDonnell, 2026." attrlink="https://mutts.com/products/strip-012621?variant=41141367603357" align="center" >}}


## Intensional and extensional definitions

In school, mathematics is generally taught by modeling a problem and solving the resulting equation:

$$ \text{Find all} \; x \in \R \; \text{such that} \; x^2 - 3x + 2 = 0 \Leftrightarrow x = 1 \; \text{or} \; x = 2.$$

We can re-write this equivalence as a set equality:

$$ \{ x \in \R : x^2 - 3x + 2 = 0 \} = \{1, 2\}.$$

In [logical reasoning](https://en.wikipedia.org/wiki/Logic), the left set is known as an **intensional definition**: elements are characterized by a common property they satisfy --- here, a quadratic equation.
By contrast, the right set is the **extensional definition**, which is the list of all elements belonging to the set.
In all rigour, when we say "solving an equation" in maths, we means turning an intensional definition into an extensional one.

Interestingly, note that while the extension form is unique, there may be multiple intensional definitions:

$$ 
\begin{aligned}
\{1, 2\} &= \{ x \in \R : x^2 - 3x + 2 = 0 \} & \quad \text{// roots of a quadratic} \\
&= \{ x \in \R : 2^x=2x \} & \quad \text{// what to say here?} \\
&= \{ n \in \N : n!=n \} & \quad \text{// fixed points of factorial} \\
&= \{ n \in \N : \exists a,b,c \in \N^*, \; a^n + b^n = c^n \} & \quad \text{// Fermat's Last Theorem} \\
&= \{ n \in \N : n \in \{1, 2\} \} & \quad \text{// ... cheating ?} \\
&= \dots
\end{aligned}
$$

Definitions by [extension and intension](https://en.wikipedia.org/wiki/Extensional_and_intensional_definitions) generalize beyond mathematics.
For example, echoing the cartoon of {{< citet "mcdonnell2026mutts" >}} in introduction, take a snowflake:

- **Intension (dictionary)**: one of the small masses in which snow commonly falls ({{< citep "oed-snowflake" >}}).
- **Intension (physics)**: a six-fold symmetric ice crystal, formed by vapor deposition around a nucleus ({{< citep "kepler1611" >}}).
- **Extension**: {❄️${}_1$, ❄️${}_2$, $\dots$} --- every individual flake of snow that has ever formed, listed one by one.

Interestingly, note that defining by intension takes one line, while the last proposition by list never ends --- which is exactly why nobody defines a snowflake by extension.
But on the other hand, we may agree that the latter is the "cleanest" view: yes it is long --- almost infinite --- but it is raw, simple, free of ambiguity, and exhaustive.

I find these concepts highly relevant to understanding what separates supervised from self-supervised learning:
**annotated/paired datasets are the extension view: simple and direct, they create an unambiguous signal that is probably the strongest possible signal for a given task** --- indeed, what could be a better signal than the explicit answer we are looking for?
But unfortunately, similarly to the near-infinite snowflake list, paired datasets are expensive to acquire: manual annotation is hard to scale, and pairs are sometimes simply impossible to acquire in nature.
Hence, I argue that **self-supervised learning is a paradigm shift: train through one or multiple properties characterizing the expected answer --- i.e. necessary and sufficient conditions (hopefully) --- and observe the result emerge**.

## The price of a weaker signal

![The tree swing cartoon](./tree_swing.avif)

On paper, intensional and extensional definitions are equivalent, but in practice, finding a set of properties that perfectly captures your target outputs is hard.
Think about it: can you name the exact properties shared by *all* natural images?
Note that this paradigm shift is not deliberate: it is imposed by the lack of ground truth data --- in an ideal world with an enormous annotated dataset, I am not sure anyone would have bothered dealing with self-supervised methods.

Therefore, it is the inexactness of any hand-crafted set of properties, combined with the sheer multiplicity of possible definitions, that creates a landscape of possible self-supervised objectives, each yielding a different training signal.
**I argue that the quality of your training signal is a direct reflection of how much your chosen properties manage to capture the ideal, unreachable extensional definition you seek.**
Nevertheless, switching to self-supervised approaches --- meaning weaker training signals --- has enabled groundbreaking advances in deep learning, as it has enabled training on orders-of-magnitude larger datasets.

Here is my own loose way to think about this tradeoff: training is a bit like estimating a quantity from i.i.d values in statistics.
If you estimator has a variance $\sigma$ for a single sample, the error shrinks roughly as $\sigma/\sqrt{N}$, where $N$ is your total sample count.
Supervised learning loss function achieves the smallest possible $\sigma$ (also known as a fully efficient estimator, one that reaches the [Cramér–Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound)): it provides the sharpest training signal possible.
By contrast, imperfect, hand-crafted, self-supervised objectives yield larger $\sigma$ but earn back an $N$ that is orders of magnitude bigger by enable larger datasets.
And empirically, that trade has paid off.
Of course, I'm deliberately oversimplifying here --- generalization, data diversity, and so on are swept under the rug --- but I think the core analogy holds.

To summarize, I think that nowadays, **building objectives for self-supervised models boils down to the art of picking the set of properties that best characterizes the target answer**.
Let's look at practical models to see these ideas in action.

## Intension signals in practice

<!-- ### Reconstruction attempts

But to be honnest ... why ? Why would reconstructing the image be a good indicator?
If you think about it, L2 reconstruction is the best signal for normal likelihood reconstruction, but other problem, there is no reason why this should be a good signal?

The good signal lives in the base of p(x)

First attempts to apply that, at the best of my kowledge, were by masked reconstruction.
The idea is simple: objects on natural images are macro, i.e they span multiple pixels: masking part of the image likely does not destroy the object, and being able to reconstruct it means the network understands it.

However, reconstruction tends not to be a good signal. By cosntruction, object can be the same and different at the same time. If I mask half of cloud for example, the right answer could be a lot of things, but it still is a cloud: there is no need to reconstruct the precise pixel-wise shape to known that is is a cloud. Hence, reconstruction is inherently a bad signal to understand scenes, because it is not aligned with the ground truth objective.
This is an idea vastly populised by LeCun in his idea: for representation learning, you loss should be in the latent, representation space.  -->

### Invariants for natural images

![Data augmentations](./augmentations.avif)

If you think about it, equations in physics act as **invariants**, in the same sense as our quadratic equation above.
They define the set of valid solutions: every valid wavefunction must satisfy the Schrödinger equation, every fluid field the Navier–Stokes equations, every moving object the laws of motion.
This idea alone is strong enough to train a neural network: physics-informed neural networks ({{< citep "raissi2019pinn" >}}) fit a network by penalizing how much it violates the governing equation, with no labeled solution required --- the equation itself is the self-supervised signal.

Can we apply the same idea to natural images?
If the task is to recognize objects, two invariants come to mind:
1. Objects are macro: they span many pixels, so cropping or zooming rarely destroys their identity.
2. Objects stay recognizable under a wide range of transformations --- flips, contrast, brightness, color shifts, etc ...

These transformations are known today as **data augmentation** ({{< citep "chen2020simple" >}}).

Numerous works confirm that these invariants shape the latent space well ({{< citep "chen2020simple" >}}, {{< citep "purushwalkam2020demystifying" >}}), so we can treat them as **necessary** conditions.
But **they are not sufficient**: a constant, collapsed representation trivially satisfies both. 
This is the **representation collapse issue** ({{< citep "chen2021simsiam" >}}), and avoiding it requires an additional property.

### Contrastive learning

![Contrastive learning](./contrastive.avif)

SimCLR ({{< citep "chen2020simple" >}}) tackles this by leveraging the size of the dataset: **two randomly picked images are likely different objects**.
Suppose the representation space is the unit sphere $\{ z \in \R^d : ||z||_2 = 1 \}$.
One can then align an image with its augmented view (built from the invariances above) by maximizing $\dotprod{\vz_1}{\vz_1'}$, and push dissimilar images apart by minimizing $\dotprod{\vz_1}{\vz_2}$.
The first term learns the invariance, the second shapes the latent space.

In practice, SimCLR replaces this toy loss with the softmax-based **InfoNCE** loss, contrasting each view against every other view in the batch at once, with a temperature $\tau$ controlling how sharply negatives are pushed apart:

$$ \Ls(\vz_1, \vz_1') = -\log \frac{\exp(\dotprod{\vz_1}{\vz_1'}/\tau)}{\sum_{k \neq 1} \exp(\dotprod{\vz_1}{\vz_k}/\tau)}. $$

Following this idea of positive-negative pairs --- known as **contrastive learning** ---, {{< citet "radford2021learning" >}} created CLIP by replacing an augmented pair $(\text{image}_1, \text{image}_1')$ with an image-caption pair $(\text{image}_1, \text{caption of image}_1)$, mapping images and text into a shared space and paving the way for modern VLMs.

A known weakness of this scheme is that some "negatives" sampled from the batch actually share the same content as the anchor; {{< citet "robinson2021contrastive" >}} address this with an importance-sampling correction that debiases these false negatives while emphasizing genuinely hard ones.

### Leverage cross-correlation

![Barlow Twins](./barlow.avif)

Let's view the embedding as a random variable $\rz = f_\theta (\rvx)$, where $\rvx$ is a random image from the dataset, and assume for now that $\rz$ is scalar.
A collapsed representation means $\rz = \cst$, i.e. $\Var(\rz) = 0$.
A natural fix is then to force the embedding to vary, e.g. by imposing $\Var(\rz) = 1$.

Barlow Twins ({{< citet "zbontar2021barlow" >}}) avoids representation collapse this way, through normalization.
Let $\rvx'$ be another view of the same image $\rvx$, and $\rz' = f_\theta (\rvx')$.
A natural objective is then to maximize the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between the two views:

$$\Ls(\rz, \rz') = 1 - \rho(\rz, \rz') = 1 - \Cov(\rz, \rz') / \sqrt{\Var(\rz)\Var(\rz')}.$$

This single term does two things at once: it pulls the two views together, and it keeps their variance from collapsing to zero.

Embeddings are usually $d$-dimensional rather than scalar (hence denoted in bold from here on, $\rvz$ and $\rvz'$). {{< citet "zbontar2021barlow" >}} add a second idea: each component of the vector should carry different information, i.e. for $i \neq j$ we want $\Cov([\rvz]_i, [\rvz']_j) = 0$.
Combined, this means pushing the whole cross-correlation matrix toward the identity:

$$\Ls(\rvz, \rvz') = \frobenius{\rmC}{\mI} \quad \text{where} \; 
[\rmC]_{i,j} = \rho([\rvz]_i, [\rvz']_j)
$$

Barlow Twins therefore folds two properties into a single matrix objective: diagonal terms pushed to 1 both for invariance and to prevent collapse, and off-diagonal terms pushed to 0 to protect against redundancy.
Note that in practice, statistics are estimated over each mini-batch independently.

VICReg ({{< citep "bardes2022vicreg" >}}) later disentangled these into three separate, explicitly weighted losses --- variance, invariance, covariance --- improving on Barlow Twins with finer control and better scalability.

### Impose the distribution

![LeJEPA](./lejepa.avif)

Another way to read Barlow Twins is as shaping the distribution of the embeddings: it constrains the distribution to have unit variance long every axis.

LeJEPA ({{< citep "balestriero2025lejepa" >}}) takes this further, and **explicitely targets a specific shape for the embeddings distribution**.
The authors show that, in the absent of any knowledge of the downstream task, **the optimal shape for the latent space is a full isotropic Gaussian**: if $\rvx$ ranges over all possible inputs, we want enforce $f_\theta(\rvx) \sim \gN(0, \mI)$.
Their loss combines an invariance term with a distribution-matching term:

$$\Ls(\rvz, \rvz') = ||\rvz - \rvz'||_2^2 + D(\rvz, \gN(0, \mI)),$$

where $D$ is an integral probability metric. 
They propose to use the Epps--Pulley normality test applied to random 1D projections of the embeddings for D --- which, to the best of my knowledge, is equivalent to the invariant-kernel MMD objective I derived in a [previous post](/posts/mmd/). 
They arguee that this estimator is trivial to parallelize, linear in complexity, and is more robust than the "first two moment-matching" approach of Barlow Twin.

You can see the same idea at work in DINOv2 ({{< citep "oquab2023dinov2" >}}) through the KoLeo regularizer, which instead targets a uniform distribution on the $d$-hyperball --- I also wrote a [full post on it](/posts/koleo_regularization/) if you want the details.

### Litteraly "self-supervised" through distillation

Finally, I would like to discuss a very different lign of work that takes the word "self-supervised learning" to the letter.

Imagine that you have pre-trained a neural network using a small annotated dataset.
Your hope is that it generalizes to unseen images.
If that is true, then maybe we could use this network to annotate a larger unlabel dataset.
And then, you could retrain another neural network based on this new large dataset!

At first glance, it seems like magic: how could additionnal information appear from this pipeline?
My understanding is that transfering knowledge from one network to another --- something called distrillation --- allows to leverage the unseen data and create a clear learning signal for the student is much stronger than the learning signal from the teacher, leading to better generalization.
To some extent, I like to think of it as a re-organization of the same latent space thanks to both more diverse data and a cleaner training signal.

That's the core idea behind BYOL ({{< citep "grill2020bootstrap" >}}), which stands for *Boostrap Your Own Latent*.
But even more than that, {{< citet "grill2020bootstrap" >}} have proposed **to differentiate this pipeline**:
instead of considering one completely pretrain network $f_{\theta_0}$, they developed a method to simulatenously train the small network and the large one.
The idea is that the annotator network --- called a teacher --- is nothing but a moving exponential average of the student.
But to make it a good teacher, they propose to 1) increase the temperature of the last sigmoid layer, which result in a sharper output vector --- we increase the confidence of the teacher in its answer, with the hope to slowly increase its confidence over time with this trick and 2) they re-center the mean output to create a well centered space and prevent collapsing issues.

The refined version of this method --- with plenty of added tricks --- is the DiNO family ({{< citep "caron2021emerging" >}}; {{< citep "oquab2023dinov2" >}}), where especially DINOv2 ({{< citep "oquab2023dinov2" >}}) showed exceptionnal performance at its release time.

## Finding good signals

After trying to untagled what makes a learning pipeline efficient, can we derive some flywheel, some bluprints from this that could hel us build better tools?
- First and foremost. Define what you want. Imagine if you had accees to the solution: what properties should it respect ? Add those which you kown you don't care about as invariant, focus on what you want.
- Prvent collapse. Check if 0 is a trivial solution: if it is, you are missing a corner case, hence you need to add somehting.
- Finalyy, I would arguee that simpe systems are usually the best ones. This is a natural way of thinking, but stacking complexity layers on top of each other tends to yield to over ingenieered. The world is beatifull, and often simple well defined objkective are available, through this is only my inner-presonnal intuition.

Have fun building you mdoels, fellow reseacrher !