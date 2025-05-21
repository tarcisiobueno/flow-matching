# Flow Matching for Generative Modeling

This repository contains an implementation of Flow Matching based on the paper Flow Matching for Generative Modeling by Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

![Comparison between Conditional Optimal Transport FM and VP Diffusion FM](/checkers_at_t.png)
![Trajectories of sample points from source to target distribution](/trajectories.png)

## Some of the theory behind Flow Matching for Generative Models:

We have a training dataset of samples from some target distribution $q$ over $\mathbb{R}$. 

To build a model capable of generating new samples from $q$, Flow Matching (FM) builds a probability path $(p_t)_{0 \leq t \leq 1}$ from a source Gaussian distribution $p_0=p$ to the data target distribution $p_1=q$. 

FM is an objective to train the velocity field neural network which is used to convert the source distribution $p_0$ into the target distribution $p_1$, along the probability path $p_t$. 

After this neural network is trained, we can generate new samples by drawing a sample from the source distribution $X_0 \sim p$ and solving the ODE determined by the velocity field. 

We do this in two steps: 

1. we design the probability path $p_t$ interpolating between $p$ and $q$.

Let the source distribution be $p:=p_0=\mathcal{N}(x|0, I)$. We construct the probability path $p_t$ known as the conditional optimal-transport or linear path:

$$
p_t(x) = \int p_{t|1}(x|x_1) q(x_1) \, dx_1
$$

where $p_{t|1}(x|x_1)=\mathcal{N}(x|tx_1, (1-t^2)^2I)$.

Based on this probability path we define the random variable $X_t \sim p_t$ by drawing $X_0$ from $p$, drawing $X_1$ from $q$, and taking their linear combination:

$$
X_t = tX_1 + (1-t)X_0 \sim p_t
$$

2. we train a velocity field $u^\theta_t$.

To do this we could regress the neural network velocity field $u^\theta_t$ to a known target velocity field $u_t$. For this, we would use the Flow Matching Loss (FML):

$$
L_{FM} = E_{t, X_t} u^\theta_t(X_t)-u_t(X_t)^2
$$

where $t \sim U[0, 1]$ and $X_t \sim p_t$.

As stated in the article, FML is intractable. So, we simplify it by conditioning the loss on a single target $X_1=x_1$ picked at random from the training set. 

$$
X_t|_1 = t x_1 + (1 - t) X_0 \quad \sim \quad p_{t|1}(\cdot | x_1) = \mathcal{N}(\cdot | t x_1, (1 - t)^2 I)
$$

We then obtain the Conditional Flow Matching Loss (CFM).

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, X_t, X_1} \| u^\theta_t(X_t) - u_t(X_t | X_1) \|^2
$$

$\text{where} \quad t \sim U[0,1], \quad X_0 \sim p, \quad X_1 \sim q$.

One important contribution given by the article was to show that FML and CFM provide the same gradients to learn $u^\theta_t$, that is $\nabla_\theta L_{FM}(\theta)=\nabla_\theta L_{CFM}(\theta)$.

Solving $\frac{d}{dt}X_{t|1} = u_t(X_{t|1}|X_1)$ and pluging the result into CFM, we have:

$$
\mathcal{L}^{\text{OT,Gauss}}_{\text{CFM}}(\theta) = \mathbb{E}_{t, X_0, X_1} \| u^\theta_t(X_t) - (X_1-X_0) \|^2
$$
