# AR1_DDM
Sven Wientjes

# Autoregressive DDM

In this document, we will discuss, fit, and evaluate a Drift Diffusion
Model (DDM) with AutoRegressive (AR1) processes for it’s main
parameters, using the software [Stan](https://mc-stan.org/cmdstanr/),
and demonstrate how this model can converge in **60 to 90 seconds** for
a typical participant[^1]. This tutorial is heavily inspired by the
great work of [Vloeberghs et
al. (2026)](https://www.biorxiv.org/content/10.64898/2026.03.20.713186v1.abstracthttps://www.biorxiv.org/content/10.64898/2026.03.20.713186v1.abstract),
who first developed this AR(1) variant of the DDM, identified a unique
signature of it in terms of *temporally clustered errors*, and provided
an amortized inference network to estimate the parameters based on
empirical data.

By implementing the model in Stan, we can get a couple of advantages
compared to the amortized implementation of Vloeberghs et al.:

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

## The AR1-DDM model

bla.

## Exploring the data

bla.

## Fitting the model

bla.

## Exploring the fit

bla.

## 

[^1]: Results obtained using an AMD Ryzen 7 7700 CPU.
