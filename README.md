# Autoregressive DDM
Sven Wientjes

In this document, we will discuss, fit, and evaluate a Drift Diffusion
Model (DDM) with AutoRegressive (AR1) processes for it’s main
parameters, using the software [Stan](https://mc-stan.org/cmdstanr/),
and demonstrate how this model can converge in **60 to 90 seconds** for
a typical participant[^1]. This tutorial is directly inspired by the
great work of [Vloeberghs et
al. (2026)](https://www.biorxiv.org/content/10.64898/2026.03.20.713186v1.abstracthttps://www.biorxiv.org/content/10.64898/2026.03.20.713186v1.abstract),
who first developed this AR(1) variant of the DDM, identified a unique
signature of it in terms of *temporally clustered errors*, and provided
an Amortized Bayesian Inference (ABI) network to estimate the parameters
based on empirical data.

By implementing the model in Stan, we can get a couple of advantages
compared to an ABI implementation:

1.  Amortized inference is not exact. Even though the parameter recovery
    *can* be highly accurate, no mathematical *guarantee* exists that we
    are sampling from the true posterior distribution. With Stan this
    guarantee exists in theory, and Stan offers excellent diagnostics
    (i.e., $\hat{R}$ and $ESS$) to help establish this in practice.

2.  Amortized inference fixes the prior distribution and number of
    trials that can be handled. Implementation in Stan allows variable
    trial numbers per participant, and allows for easy changes of the
    prior distribution.

3.  The implementation we will consider has a principled way of dealing
    with missing data. Missing data is common in choice-RT experiments,
    because often fast and slow outliers are excluded. These data are
    assumed not to arise from hypothesized evidence accumulation
    dynamics, but from contaminating processes.

This tutorial is interactive. You can clone or download this repository
and open the `MovingDDM.Rproj` to run code yourself.

## The AR1-DDM model

The DDM is a generative model of binary responses and related response
times. It assumes that within a trial, evidence accumulates with an
average rate of $\delta$ (the *drift rate*) until an upper or lower
boundary is reached. These boundaries are separated by a parameter $a$.
The evidence accumulation process does not have to start in the middle
between the upper and lower boundary–it can start closer to either
boundary, relatively parameterized as $\beta \in [0,1]$. The observed
response times also incorporate non-decision time $\tau$, on top of the
time it takes to complete the evidence accumulation process. Under this
standard formulation of the DDM, the probability of hitting the upper
boundary at time $y$ can be computed with the [Wiener first passtime
distribution](https://mc-stan.org/docs/functions-reference/positive_lower-bounded_distributions.html#wiener-first-passage-time-distribution):

$\frac{\alpha3}{(y-\tau)^{3/2}} e^{-\delta \alpha \beta - \frac{\delta^2(y-\tau)}{2}}\sum_{k=-\infty}^\infty (2k+\beta)\phi(\frac{2k\alpha+\beta}{\sqrt{y-\tau}})$

## Exploring the data

Lets set up the packages and functions we will use in this tutorial
first:

``` r
library(cmdstanr)
library(data.table)
library(ggplot2)
library(RWiener)
source("functions/CAF.R")
source("functions/correlated_errors.R")
```

Like Vloeberghs et al., we will use the data of Experiment 2B from
[Desender et al. (2022)](https://doi.org/10.1038/s41467-022-31727-0).
Let’s load it and do some simple data wrangling:

``` r
# Load data
MyData <- fread("https://raw.githubusercontent.com/kdesende/dynamic_influences_on_static_measures/refs/heads/main/data_exp2B.csv")

# Get nicer participant numbers
MyData$pp         <- ordered(MyData$sub)
levels(MyData$pp) <- c(1:99)

# Recode stimulus 0 to -1
MyData[stim==0,stim:=-1]

# Mark trials for exclusion
MyData$yi <- 1
MyData[rt<0.1|rt>3.0,yi:=0]
```

Note that we use `$yi` to mark fast and slow outliers with a 0, while
all other trials are marked with a 1. This is how we will deal with data
exclusion, without breaking the temporal structure of the data.

### Visualizing RT distributions

We can visualize the (marginal) RT distribution of a specific
participant:

``` r
ggplot(MyData[pp==69 & yi==1], aes(x=rt)) +
  geom_density(linetype="dashed",linewidth=1) +
  theme_minimal()
```

![](README_files/figure-commonmark/unnamed-chunk-3-1.png)

### Visualizing the Conditional Accuracy Function (CAF)

The Conditional Accuracy Function (CAF) partitions this response time
distribution into different regions with equal numbers of trials in
them. It then visualizes the proportion of correct trials in each bin
against the mean RT of the trials in that bin. This reveals
substantially more errors for very fast responses, as well as
substantially more errors for very slow responses. The typical DDM with
4 parameters can not explain these fast and slow errors–it predicts a
relatively uniform relationship between choice and accuracy.

``` r
# Get CAF and confidence interval
CAF <- calculate_group_caf(MyData[yi==1], "rt", "cor", "pp", num_bins=7)
CAF[,min_acc:=mean_acc-1.96*se_acc]
CAF[,max_acc:=mean_acc+1.96*se_acc]

## Group conditional accuracy function
ggplot(CAF, aes(x=mean_rt, y=mean_acc)) +
  geom_point() +
  geom_errorbar(aes(ymin=min_acc,ymax=max_acc),width=0.05) +
  geom_line() +
  theme_minimal() +
  xlab("RT") + ylab("Accuracy")
```

![](README_files/figure-commonmark/unnamed-chunk-4-1.png)

### Visualizing correlated errors

Now for the signature of the autocorrelated DDM, we can assign each
trial into the 50% fastest or 50% slowest errors. We can then visualize
for each error class, what was the probability of the preceding error
classes. This reveals that errors are likely to repeat both in identity
(i.e., an error by responding with `-1` is more likely to be followed by
another error responding with `-1`) as well as by speed (i.e., slow
errors tend to occur successively, and fast errors tend to occur
successively). A typical 4-parameter DDM treats each trial as
independent and identically distributed. This means that preceding
errors bear no relation to successive error responses by definition.

``` r
group_stats <- get_error_cor(MyData)
ggplot(group_stats, aes(x = er_class, y = mean_prob, fill = prev_er_class)) +
  geom_bar(
    stat     = "identity", 
    position = position_dodge(width = 0.9), 
    color    = "black", 
    lwd = 0.3
  ) +
  geom_errorbar(
    aes(ymin     = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.9),
    width    = 0.25, # Width of the error bar caps
    color    = "grey30"
  ) +
  scale_fill_brewer(palette = "Set2", name = "Previous Error Class") +
  theme_minimal() +
  xlab("Error class") + ylab("Probability of previous error class") +
  theme(legend.position="top")
```

![](README_files/figure-commonmark/unnamed-chunk-5-1.png)

## Fitting the model

Now let’s fit the autocorrelated DDM to the empirical data using Stan.
The model code is contained in `models/DDM_AR1.stan`. To make this model
run fast and stable, we can reparameterize the local fluctuations as
follows:

Remember that an autoregressive process evolves according to the
equation

$X_t = \rho X_{t-1} + \epsilon_{t}$ with
$\epsilon_{\gamma, t} \sim  \mathcal{N}(0, \sigma^2)$

but this formulation yields a total stationary variance of
$Var(X) = \frac{\sigma^2}{1-\rho^2}$. This means that the variance
changes unpredictably whenever $\rho$ or $\sigma^2$ change. Stan does
not like these kind of dependencies, but luckily we can separate them.

We can fix the variance of the latent state $X_t$ over time to be
exactly 1. We can do this by drawing standard normal innovations
$Z_t \sim \mathcal{N}(0,1)$ and scaling them by $\sqrt{1-\rho^2}$ :

$X_t = \rho X_{t-1} + \sqrt{1-\rho^2} Z_t$

We can then make whatever parameter we want dependent on this
standardized latent state, and include a separate parameter $\sigma$ to
scale this latent state. For example, for the drift rates:

$\delta_t = \delta + \sigma_\delta \gamma_{t}$

where now the fluctuations of drift rate $\gamma_t$ are implemented the
same way we described the dynamics above for $X_t$. This
reparameterization separates the influence of correlation $\rho$ and
scale $\sigma$, allowing for **much** faster and more stable sampling in
Stan.

### Actually fitting the model

Bla.

## Exploring the fit

bla.

## 

[^1]: Results obtained using an AMD Ryzen 7 7700 CPU.
