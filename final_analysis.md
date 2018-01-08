Final study analysis
================

-   [Introduction](#introduction)
    -   [Libraries required for this analysis](#libraries-required-for-this-analysis)
-   [Load and clean data](#load-and-clean-data)
-   [Naive learning curves](#naive-learning-curves)
    -   [By vis](#by-vis)
    -   [By arrival](#by-arrival)
-   [Bayesian beta regression](#bayesian-beta-regression)
    -   [Model](#model)
    -   [Priors](#priors)
    -   [Fit model](#fit-model)
-   [Posterior prediction](#posterior-prediction)
-   [Model-based learning curves](#model-based-learning-curves)
    -   [Trends in mean](#trends-in-mean)
    -   [Trends in standard deviation](#trends-in-standard-deviation)
-   [Performance on last trial](#performance-on-last-trial)
    -   [Mean](#mean)
    -   [Standard deviation](#standard-deviation)

Introduction
------------

This document contains the final analysis of expected / optimal payoffs from the paper.

### Libraries required for this analysis

If you have not installed RStan previously, **install it before installing any other packages below**. Rstan can be finicky to install, so we recommend carefully following the [RStan Getting Started Guide](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started). **Do not** simply issue ~~`install.packages("rstan")`~~, as this may result in a non-working RStan installation.

Once RStan is installed, you can set up the `tidybayes` package as follows:

``` r
install.packages("devtools")
devtools::install_github("mjskay/tidybayes")
```

Finally, install any remaining packages from the list below that you do not already have using `install.packages(c("package1", "package2", ...))`. The `import::from(packagename, function)` syntax requires the `import` package to be installed.

``` r
library(magrittr)
library(stringi)
library(tidyverse)
library(modelr)
library(forcats)
library(snakecase)
library(lsmeans)
library(broom)
library(lme4)
library(dotwhisker)
library(directlabels)
library(rstan)
library(brms)
library(rlang)
library(tidybayes)
library(cowplot)
library(RColorBrewer)
library(bindrcpp)
import::from(gamlss.dist, qBCT)
import::from(boot, logit)
```

Load and clean data
-------------------

First, let's read in the data, do some basic name and datatype cleaning:

``` r
df = read.csv("data/final_trials.csv") %>%
  as_tibble()

head(df)
```

| vis   | arrival | participant | distribution | scenario |  trial|        mu|      sigma|         nu|       tau|  mode|  duration|  response|  payoff|  expected\_payoff|  optimal\_payoff|  expected\_over\_optimal|
|:------|:--------|:------------|:-------------|:---------|------:|---------:|----------:|----------:|---------:|-----:|---------:|---------:|-------:|-----------------:|----------------:|------------------------:|
| dot50 | FALSE   | p369        | d060         | s3       |      0|  34.43021|  0.1038757|  -7.673986|  1.969532|    18|  22.85116|        16|    2151|          1857.236|         1987.309|                0.9345483|
| dot50 | FALSE   | p369        | d036         | s3       |      1|  36.11904|  0.1151801|  -3.047448|  2.024402|    20|   8.40406|        18|    2031|          1743.631|         1916.901|                0.9096093|
| dot50 | FALSE   | p369        | d001         | s3       |      2|  24.69326|  0.0507094|  -1.711638|  1.627509|    10|  17.24604|         9|    2078|          1601.720|         1949.349|                0.8216692|
| dot50 | FALSE   | p369        | d001         | s3       |      3|  24.69326|  0.0507094|  -1.711638|  1.627509|    10|  29.20262|         9|    2095|          1601.720|         1949.349|                0.8216692|
| dot50 | FALSE   | p369        | d106         | s3       |      4|  30.73842|  0.0813301|  -2.877004|  1.845427|    15|  12.57680|        14|    2084|          1722.179|         1944.729|                0.8855624|
| dot50 | FALSE   | p369        | d067         | s3       |      5|  24.73388|  0.0508896|  -4.638728|  1.629045|    10|  11.80909|         8|    2070|          1839.135|         1967.857|                0.9345875|

The set of conditions looks like this:

``` r
df %>%
  group_by(vis, arrival) %>%
  summarise(n(), length(unique(participant)))
```

| vis         | arrival |   n()|  length(unique(participant))|
|:------------|:--------|-----:|----------------------------:|
| cdf         | FALSE   |  1092|                           25|
| cdf         | TRUE    |   599|                           15|
| dot20       | FALSE   |   685|                           17|
| dot20       | TRUE    |   734|                           18|
| dot50       | FALSE   |   635|                           16|
| dot50       | TRUE    |   705|                           17|
| interval    | FALSE   |   864|                           21|
| interval    | TRUE    |   793|                           18|
| none        | FALSE   |   651|                           16|
| none        | TRUE    |   747|                           18|
| pdf         | FALSE   |   705|                           17|
| pdf         | TRUE    |  3840|                           57|
| pdfinterval | FALSE   |   605|                           15|
| pdfinterval | TRUE    |   750|                           18|
| text        | FALSE   |   602|                           15|
| text        | TRUE    |   827|                           20|
| text60      | FALSE   |   647|                           16|
| text60      | TRUE    |   768|                           19|
| text99      | FALSE   |   537|                           13|
| text99      | TRUE    |  2616|                           41|

We will add `trial_normalized`, which will go from -0.5 (the first trial) to 0.5 (the last trial) to help model convergence.

``` r
max_trial = 39
stopifnot(max(df$trial) == max_trial)

df %<>%
  mutate(
    trial_normalized = ((trial - max_trial) / max_trial) + 0.5
  )
```

We will also add the intervals that participants would have been shown in the text conditions:

``` r
df %<>%
  mutate(
    text_interval = round(gamlss.dist::qBCT(1 - .85, mu, sigma, nu, tau) - 15),
    text60_interval = round(gamlss.dist::qBCT(1 - .60, mu, sigma, nu, tau) - 15),
    text99_interval = round(gamlss.dist::qBCT(1 - .99, mu, sigma, nu, tau) - 15)
  )
```

We'll also exclude scenario 4 for now, since it would need to be analyzed separately (because it is binary choice) and because we did not pre-register a model for it:

``` r
df %<>%
  filter(scenario != "s4") %>%
  mutate(scenario = factor(scenario))
```

Finally, for the beta regression models, we will need `expected_over_optimal` to be guaranteed to be between 0 and 1 (exclusive) --- currently it is between 0 and 1 (inclusive). So we'll create an "adjusted" normalized response that is guaranteed to be between 0 and 1 (exclusive). This just adjusts values that would be 1 downward slightly:

``` r
df %<>%
  mutate(
    expected_over_optimal_adjusted = ifelse(expected_payoff == optimal_payoff, 
      optimal_payoff / (optimal_payoff + 1),
      expected_payoff / optimal_payoff
    )
  )
```

This transformation does not substantially change responses, but does allow us to use the logit transformation on them (or use the beta distribution to model them, as we will see shortly). You can see that the distributions are essentially identical (black is original, red is adjusted):

``` r
df %>%
  filter(trial > 35) %>% #just look at the last couple of trials
  ggplot(aes(x = expected_over_optimal)) +
  stat_density() +
  stat_density(aes(x = expected_over_optimal_adjusted), fill = NA, color = "red") +
  facet_wrap(~ vis)
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-9-1.png)

These plots show distributions of responses in each condition in the last 5 trials. This also suggests that beta regression might be an appropriate choice, since the conditional distributions are beta-like.

An excerpt from the cleaned dataset:

``` r
head(df)
```

| vis   | arrival | participant | distribution | scenario |  trial|        mu|      sigma|         nu|       tau|  mode|  duration|  response|  payoff|  expected\_payoff|  optimal\_payoff|  expected\_over\_optimal|  trial\_normalized|  text\_interval|  text60\_interval|  text99\_interval|  expected\_over\_optimal\_adjusted|
|:------|:--------|:------------|:-------------|:---------|------:|---------:|----------:|----------:|---------:|-----:|---------:|---------:|-------:|-----------------:|----------------:|------------------------:|------------------:|---------------:|-----------------:|-----------------:|----------------------------------:|
| dot50 | FALSE   | p369        | d060         | s3       |      0|  34.43021|  0.1038757|  -7.673986|  1.969532|    18|  22.85116|        16|    2151|          1857.236|         1987.309|                0.9345483|         -0.5000000|              16|                18|                12|                          0.9345483|
| dot50 | FALSE   | p369        | d036         | s3       |      1|  36.11904|  0.1151801|  -3.047448|  2.024402|    20|   8.40406|        18|    2031|          1743.631|         1916.901|                0.9096093|         -0.4743590|              17|                20|                 9|                          0.9096093|
| dot50 | FALSE   | p369        | d001         | s3       |      2|  24.69326|  0.0507094|  -1.711638|  1.627509|    10|  17.24604|         9|    2078|          1601.720|         1949.349|                0.8216692|         -0.4487179|               8|                 9|                 2|                          0.8216692|
| dot50 | FALSE   | p369        | d001         | s3       |      3|  24.69326|  0.0507094|  -1.711638|  1.627509|    10|  29.20262|         9|    2095|          1601.720|         1949.349|                0.8216692|         -0.4230769|               8|                 9|                 2|                          0.8216692|
| dot50 | FALSE   | p369        | d106         | s3       |      4|  30.73842|  0.0813301|  -2.877004|  1.845427|    15|  12.57680|        14|    2084|          1722.179|         1944.729|                0.8855624|         -0.3974359|              13|                15|                 6|                          0.8855624|
| dot50 | FALSE   | p369        | d067         | s3       |      5|  24.73388|  0.0508896|  -4.638728|  1.629045|    10|  11.80909|         8|    2070|          1839.135|         1967.857|                0.9345875|         -0.3717949|               8|                 9|                 4|                          0.9345875|

Naive learning curves
---------------------

### By vis

Let's start with a high-level view (and a somewhat poor model: linear without accounting for participant or scenario) to see how people improve over the course of the trials in the different conditions:

``` r
df %>%
  ggplot(aes(x = trial, y = expected_over_optimal)) +
  stat_summary(fun.data = mean_se) +
  geom_hline(yintercept = 1) +
  stat_smooth(method = lm) +
  facet_wrap(~ vis)
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-11-1.png)

### By arrival

The other question is if providing the arrival information at the right makes a difference.

``` r
df %>%
  ggplot(aes(x = trial, y = expected_over_optimal, color = arrival)) +
  stat_summary(fun.data = mean_se, position = position_dodge(width = .5)) +
  geom_hline(yintercept = 1) +
  stat_smooth(method = loess) 
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-12-1.png)

It doesn't seem to make a huge difference, so for simplicity we will pool the arrival variants. We saw something similar in the pilot analysis, and a pooled model for arrival is what we [pre-registered](pre-registration.pdf).

Bayesian beta regression
------------------------

One of the displeasing aspects of applying a linear model here is that it won't work well for prediction --- the data is skewed and bounded above at 1. Since normalized responses will always be between 0 and 1, a logit transformation might help here. This would also ensure that learning curves can approach -- but never pass --- optimal, which is a property we should expect them to have.

### Model

To build a model for these data, we'll use a beta regression, which is bounded on (0,1), and uses a logit link. The model and priors here are also what we [pre-registered](pre-registration.pdf).

### Priors

These are the priors we can set:

``` r
get_prior(bf(
    expected_over_optimal_adjusted ~ trial_normalized + (trial_normalized|participant) + (1|scenario), 
    phi ~ vis*trial_normalized
  ),
  data = df, family = Beta)
```

| prior                | class     | coef                             | group       | resp | dpar | nlpar | bound |
|:---------------------|:----------|:---------------------------------|:------------|:-----|:-----|:------|:------|
|                      | b         |                                  |             |      |      |       |       |
|                      | b         | trial\_normalized                |             |      |      |       |       |
| lkj(1)               | cor       |                                  |             |      |      |       |       |
|                      | cor       |                                  | participant |      |      |       |       |
| student\_t(3, 0, 10) | Intercept |                                  |             |      |      |       |       |
| student\_t(3, 0, 10) | sd        |                                  |             |      |      |       |       |
|                      | sd        |                                  | participant |      |      |       |       |
|                      | sd        | Intercept                        | participant |      |      |       |       |
|                      | sd        | trial\_normalized                | participant |      |      |       |       |
|                      | sd        |                                  | scenario    |      |      |       |       |
|                      | sd        | Intercept                        | scenario    |      |      |       |       |
|                      | b         |                                  |             |      | phi  |       |       |
|                      | b         | trial\_normalized                |             |      | phi  |       |       |
|                      | b         | visdot20                         |             |      | phi  |       |       |
|                      | b         | visdot20:trial\_normalized       |             |      | phi  |       |       |
|                      | b         | visdot50                         |             |      | phi  |       |       |
|                      | b         | visdot50:trial\_normalized       |             |      | phi  |       |       |
|                      | b         | visinterval                      |             |      | phi  |       |       |
|                      | b         | visinterval:trial\_normalized    |             |      | phi  |       |       |
|                      | b         | visnone                          |             |      | phi  |       |       |
|                      | b         | visnone:trial\_normalized        |             |      | phi  |       |       |
|                      | b         | vispdf                           |             |      | phi  |       |       |
|                      | b         | vispdf:trial\_normalized         |             |      | phi  |       |       |
|                      | b         | vispdfinterval                   |             |      | phi  |       |       |
|                      | b         | vispdfinterval:trial\_normalized |             |      | phi  |       |       |
|                      | b         | vistext                          |             |      | phi  |       |       |
|                      | b         | vistext:trial\_normalized        |             |      | phi  |       |       |
|                      | b         | vistext60                        |             |      | phi  |       |       |
|                      | b         | vistext60:trial\_normalized      |             |      | phi  |       |       |
|                      | b         | vistext99                        |             |      | phi  |       |       |
|                      | b         | vistext99:trial\_normalized      |             |      | phi  |       |       |
| student\_t(3, 0, 10) | Intercept |                                  |             |      | phi  |       |       |

We'll set weakly-informed priors for the various classes of coefficients, as per [our pre-registration](http://aspredicted.org/blind.php?x=g2yb2f):

``` r
pr_beta = c(
  prior(normal(0, 1), class = b),
  # these prior intercepts are wide and cover 0 (50% on the logit scale), but
  # also assume some likely better-than-50% performance on average --- this
  # was chosen to aid convergence during the pilot, but does not have a strong
  # impact on final estimates.
  prior(normal(2, 2), class = Intercept),
  prior(normal(2, 2), class = Intercept, dpar = phi),
  prior(normal(0, 1), class = b, dpar = phi),
  prior(student_t(3, 0, 1), class = sd)
)
```

### Fit model

Let's fit the model:

``` r
warmup = 2000
iter = warmup + 2000
thin = 2

mbeta = brm(bf(
    expected_over_optimal_adjusted ~ vis*trial_normalized + (trial_normalized|participant) + (1|scenario), 
    phi ~ vis*trial_normalized),
  data = df, prior = pr_beta, 
  control = list(adapt_delta = 0.9995, max_treedepth = 15, stepsize = 0.005),
  warmup = warmup, iter = iter, thin = thin,
  family = Beta
  )
```

Note that the above model takes a long time to run (about 24 hours), primarily because we are estimating a random effect for a factor with only three groups (scenario)---See Gelman (2006), *Prior distributions for variance parameters in hierarchical models* for more discussion of this issue. In retrospect we should have pre-registered a zero-avoiding prior for variance hyperpriors (a model with a non-zero-centered tighter truncated t prior only takes an hour or two to fit and yields relatively similar estimates), but we pre-registered the above priors, so we'll stick to them. To save time we'll just load the fitted model from disk:

``` r
mbeta = read_rds("models/final_model.rds")
```

Some model diagnostics (note the funnel shape on `sd_scenario__Intercept`, this is why our model takes awhile to fit):

``` r
pairs(mbeta$fit, pars = c("b_Intercept", "b_trial_normalized", "sd_participant__Intercept",  "sd_participant__trial_normalized", 
  "sd_scenario__Intercept",  
  "cor_participant__Intercept__trial_normalized", "b_phi_Intercept", "b_phi_trial_normalized"))
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-17-1.png)

Posterior prediction
--------------------

We'll start with posterior predictions. First we'll decide on a display order for consistency across charts:

``` r
vis_display_order = c("dot50", "cdf", "dot20", "text99", "text60", "pdfinterval", "pdf", "interval", "text", "none")
```

Then we'll generate posterior predictions:

``` r
pred_beta = 
  df %>%
  data_grid(
    vis,
    trial_normalized = seq_range(trial_normalized, n = 20)
  ) %>%
  add_predicted_samples(mbeta, re_formula = NULL, allow_new_levels = TRUE) %>%    
  mean_qi(.prob = c(.95, .8, .5))
```

And plot them, along with data (note that some conditions, notably `pdf`, had more data collected):

``` r
# shared properties of posterior prediction and fit line plots
fit_line_plot_settings = list(
  scale_x_continuous(breaks = seq(-0.5, 0.5, length.out = 40), labels = c("1", rep("", 38), "40")),
  coord_cartesian(expand = FALSE),
  xlab("Trial"),
  theme(
    panel.grid = element_blank(), panel.spacing.x = unit(5, "points"),
    strip.background = element_blank(), strip.text = element_text(hjust = 0.5, color = "black"),
    axis.title.x = element_text(hjust = 0)
  ))

post_pred_plot = pred_beta %>%
  ungroup() %>%
  mutate(vis = fct_relevel(vis, vis_display_order)) %>%
  ggplot(aes(x = trial_normalized)) +
  geom_lineribbon(aes(y = pred), data = pred_beta) +
  geom_hline(yintercept = 1) +
  scale_fill_brewer(guide = guide_legend(reverse = TRUE)) +
  geom_hline(yintercept = seq(.4, .95, by=.1), color="gray75", alpha = 0.5) +
  fit_line_plot_settings + 
  facet_grid(. ~ vis) +
  ylab("Performance / Optimal strategy")
post_pred_plot +
  geom_point(aes(y = expected_over_optimal), alpha = 0.05, data = df) 
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-20-1.png)

Above, the red line is the predicted median, and the blue bands are predictive intervals for the data. We can see that most conditions get better both in terms of bias and variance over time: people get closer to optimal, and variance in performance decreases (people get more consistent). However, there are some differences, and it is hard to tell how reliable those differences are just by looking at predictions. So, let's look at posteriors for the mean and precision of estimates according to the model.

Model-based learning curves
---------------------------

First we'll generate samples of fit lines for conditional means and variances. We will use these to plot fit lines and to generate estimates for performance in the last trial. These estimates will be for the "average" scenario and "average" person:

``` r
fit_lines = df %>%
  data_grid(
    vis,
    trial_normalized = seq_range(trial_normalized, n = 20)
  ) %>%
  add_fitted_samples(mbeta, re_formula = NA, var = "mu") %>%
  ungroup() %>%
  mutate(vis = fct_relevel(vis, vis_display_order))
```

The estimates of *ϕ* (`phi`, the precision parameter of the beta distribution) are a little hard to interpret, so instead we'll derive a posterior distribution for standard deviation. We can use the fact that the standard deviation *σ* of a Beta distribution is:

$$
\\sigma = \\sqrt{\\frac{\\mu (1 - \\mu)}{(1 + \\phi)}}
$$

Thus we can transform samples from the distribution of *μ* (`mu`) and *ϕ* (`phi`) into samples from the distribution of *σ* (`sd`):

``` r
fit_lines %<>%
  mutate(sd = sqrt(mu * (1 - mu) / (1 + phi)))
```

### Trends in mean

Estimates of the mean for the "average" scenario and "average" person:

``` r
scale_fill_fit_lines = scale_fill_manual(
  values = RColorBrewer::brewer.pal(4, "Greys")[-1], guide = guide_legend(reverse = TRUE)
)

mu_lines_plot = fit_lines %>%
  ggplot(aes(x = trial_normalized, y = mu)) +
  stat_lineribbon(.prob = c(.95, .8, .5)) +
  geom_hline(yintercept = seq(.8, 1, by=.05), color="gray75", alpha = 0.5) +
  facet_grid(. ~ vis) +
  scale_fill_fit_lines +
  fit_line_plot_settings
mu_lines_plot
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-23-1.png)

### Trends in standard deviation

Estimates of the standard deviation for the "average" scenario and "average" person:

``` r
sd_lines_plot = fit_lines %>%
  ggplot(aes(x = trial_normalized, y = sd)) +
  stat_lineribbon(.prob = c(.95, .8, .5)) +
  geom_hline(yintercept = seq(0, .16, by=.04), color="gray75", alpha = 0.5) +
  facet_grid(. ~ vis) +
  scale_fill_fit_lines +
  fit_line_plot_settings
sd_lines_plot
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-24-1.png)

Performance on last trial
-------------------------

The above trend lines let us see more clearly how performance evolved as people learned. But how did people perform in the last trial? Let's get estimates of the mean and standard deviation for the "average" person in the "average" scenario on the last trial:

``` r
last_trial = df %>%
  data_grid(
    vis,
    trial_normalized = 0.5  # because we normalized trial to be from -0.5 to 0.5
  ) %>%
  add_fitted_samples(mbeta, re_formula = NA, var = "mu") %>%
  ungroup() %>%
  mutate(vis = fct_rev(fct_relevel(vis, vis_display_order))) %>%
  mutate(sd = sqrt(mu * (1 - mu) / (1 + phi)))
```

### Mean

Conditional means (for "average" person on "average" scenario) on last trial:

``` r
plot_means = last_trial %>%
  ggplot(aes(y = vis, x = mu)) +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  coord_cartesian(xlim = c(0.85, 1.0), ylim = c(1, 10.5)) 
plot_means
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-27-1.png)

Let's look at differences between each condition and the control (no uncertainty) and the best-performing (dot50) conditions.

Difference to control:

``` r
plot_means_vs_text = last_trial %>%
  mutate(vis = fct_relevel(vis, "none")) %>%
  compare_levels(mu, by = vis, comparison = control) %>%
  ungroup() %>%
  mutate(vis = factor(vis, levels = c("none - none", levels(vis)))) %>%
  bind_rows(data_frame(vis = factor("none - none", levels = levels(.$vis)), mu = 0)) %>%
  ggplot(aes(y = vis, x = mu)) +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  coord_cartesian(xlim = c(-0.05, 0.09), ylim = c(1, 10.5))
plot_means_vs_text
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-28-1.png)

Difference to dot50:

``` r
plot_means_vs_dot50 = last_trial %>%
  mutate(vis = fct_relevel(vis, "dot50")) %>%
  compare_levels(mu, by = vis, comparison = control) %>%
  ungroup() %>%
  mutate(vis = factor(vis, levels = c(levels(vis), "dot50 - dot50"))) %>%
  bind_rows(data_frame(vis = factor("dot50 - dot50", levels = levels(.$vis)), mu = 0)) %>%
  ggplot(aes(y = vis, x = mu)) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  coord_cartesian(xlim = c(-0.11, 0.01), ylim = c(1, 10.5))
plot_means_vs_dot50
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-29-1.png)

For completeness, here is all pairwise differences in means on the last trial:

``` r
last_trial %>%
  compare_levels(mu, by = vis, comparison = pairwise) %>%
  ggplot(aes(y = vis, x = mu)) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  coord_cartesian()
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-30-1.png)

And as textual intervals:

``` r
last_trial %>%
  compare_levels(mu, by = vis, comparison = pairwise) %>%
  median_qi()
```

| vis                    |          mu|    conf.low|  conf.high|  .prob|
|:-----------------------|-----------:|-----------:|----------:|------:|
| text - none            |  -0.0101293|  -0.0508042|  0.0261662|   0.95|
| interval - none        |   0.0019492|  -0.0367437|  0.0372953|   0.95|
| pdf - none             |   0.0119112|  -0.0119375|  0.0463781|   0.95|
| pdfinterval - none     |   0.0178383|  -0.0096189|  0.0529789|   0.95|
| text60 - none          |   0.0197632|  -0.0057372|  0.0558202|   0.95|
| text99 - none          |   0.0232412|  -0.0010584|  0.0585172|   0.95|
| dot20 - none           |   0.0242469|  -0.0005829|  0.0600908|   0.95|
| cdf - none             |   0.0312723|   0.0071530|  0.0683867|   0.95|
| dot50 - none           |   0.0443898|   0.0214450|  0.0832301|   0.95|
| interval - text        |   0.0119831|  -0.0256268|  0.0540434|   0.95|
| pdf - text             |   0.0223543|  -0.0052890|  0.0616844|   0.95|
| pdfinterval - text     |   0.0280364|  -0.0031028|  0.0704213|   0.95|
| text60 - text          |   0.0299372|  -0.0009649|  0.0713668|   0.95|
| text99 - text          |   0.0336868|   0.0052421|  0.0764226|   0.95|
| dot20 - text           |   0.0347286|   0.0054693|  0.0787502|   0.95|
| cdf - text             |   0.0414628|   0.0137816|  0.0866828|   0.95|
| dot50 - text           |   0.0547529|   0.0253030|  0.1049426|   0.95|
| pdf - interval         |   0.0100887|  -0.0160751|  0.0447941|   0.95|
| pdfinterval - interval |   0.0162713|  -0.0136067|  0.0539405|   0.95|
| text60 - interval      |   0.0178362|  -0.0111308|  0.0555273|   0.95|
| text99 - interval      |   0.0213810|  -0.0038926|  0.0592248|   0.95|
| dot20 - interval       |   0.0221181|  -0.0046657|  0.0603924|   0.95|
| cdf - interval         |   0.0296165|   0.0047614|  0.0675618|   0.95|
| dot50 - interval       |   0.0429808|   0.0170976|  0.0845609|   0.95|
| pdfinterval - pdf      |   0.0060471|  -0.0167498|  0.0283395|   0.95|
| text60 - pdf           |   0.0076840|  -0.0147366|  0.0309136|   0.95|
| text99 - pdf           |   0.0113386|  -0.0075102|  0.0329685|   0.95|
| dot20 - pdf            |   0.0122903|  -0.0077478|  0.0359037|   0.95|
| cdf - pdf              |   0.0194605|   0.0015843|  0.0429793|   0.95|
| dot50 - pdf            |   0.0323658|   0.0148170|  0.0602869|   0.95|
| text60 - pdfinterval   |   0.0017334|  -0.0228791|  0.0275265|   0.95|
| text99 - pdfinterval   |   0.0051980|  -0.0165034|  0.0304231|   0.95|
| dot20 - pdfinterval    |   0.0064641|  -0.0163085|  0.0318858|   0.95|
| cdf - pdfinterval      |   0.0133484|  -0.0065570|  0.0400615|   0.95|
| dot50 - pdfinterval    |   0.0261410|   0.0080377|  0.0551688|   0.95|
| text99 - text60        |   0.0033922|  -0.0172474|  0.0270297|   0.95|
| dot20 - text60         |   0.0045383|  -0.0177236|  0.0289607|   0.95|
| cdf - text60           |   0.0112517|  -0.0073555|  0.0360511|   0.95|
| dot50 - text60         |   0.0242408|   0.0066760|  0.0529506|   0.95|
| dot20 - text99         |   0.0010803|  -0.0208510|  0.0220239|   0.95|
| cdf - text99           |   0.0079059|  -0.0098663|  0.0291488|   0.95|
| dot50 - text99         |   0.0208964|   0.0050789|  0.0450512|   0.95|
| cdf - dot20            |   0.0066960|  -0.0112718|  0.0291389|   0.95|
| dot50 - dot20          |   0.0199756|   0.0036677|  0.0451732|   0.95|
| dot50 - cdf            |   0.0130902|  -0.0010975|  0.0315447|   0.95|

### Standard deviation

Conditional standard deviation (for "average" person on "average" scenario) on last trial:

``` r
plot_sd = last_trial %>%
  ggplot(aes(y = vis, x = sd)) +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  coord_cartesian(xlim = c(0, .12), ylim = c(1, 10.5))
plot_sd
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-32-1.png)

Difference to control:

``` r
plot_sd_vs_text = last_trial %>%
  mutate(vis = fct_relevel(vis, "none")) %>%
  compare_levels(sd, by = vis, comparison = control) %>%
  ungroup() %>%
  mutate(vis = factor(vis, levels = c("none - none", levels(vis)))) %>%
  bind_rows(data_frame(vis = factor("none - none", levels = levels(.$vis)), sd = 0)) %>%
  ggplot(aes(y = vis, x = sd)) +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  coord_cartesian(ylim = c(1, 10.5), xlim = c(-.06, 0.05))
plot_sd_vs_text
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-33-1.png)

Difference to dot50:

``` r
plot_sd_vs_dot50 = last_trial %>%
  mutate(vis = fct_relevel(vis, "dot50")) %>%
  compare_levels(sd, by = vis, comparison = control) %>%
  ungroup() %>%
  mutate(vis = factor(vis, levels = c(levels(vis), "dot50 - dot50"))) %>%
  bind_rows(data_frame(vis = factor("dot50 - dot50", levels = levels(.$vis)), sd = 0)) %>%
  ggplot(aes(y = vis, x = sd)) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  coord_cartesian(xlim = c(0, .085), ylim = c(1, 10.5))
plot_sd_vs_dot50
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-34-1.png)

For completeness, here is all pairwise differences in SDs on the last trial:

``` r
last_trial %>%
  compare_levels(sd, by = vis, comparison = pairwise) %>%
  ggplot(aes(y = vis, x = sd)) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  geom_halfeyeh(fun.data = median_qih, fatten.point = 1.3) +
  coord_cartesian()
```

![](final_analysis_files/figure-markdown_github/unnamed-chunk-35-1.png)

And as textual intervals:

``` r
last_trial %>%
  compare_levels(sd, by = vis, comparison = pairwise) %>%
  median_qi()
```

| vis                    |          sd|    conf.low|   conf.high|  .prob|
|:-----------------------|-----------:|-----------:|-----------:|------:|
| text - none            |   0.0204108|  -0.0014530|   0.0436031|   0.95|
| interval - none        |   0.0059013|  -0.0143301|   0.0287323|   0.95|
| pdf - none             |   0.0031607|  -0.0139023|   0.0187764|   0.95|
| pdfinterval - none     |  -0.0145964|  -0.0325773|   0.0018895|   0.95|
| text60 - none          |  -0.0141391|  -0.0320790|   0.0020835|   0.95|
| text99 - none          |  -0.0128095|  -0.0302626|   0.0024942|   0.95|
| dot20 - none           |  -0.0259783|  -0.0441025|  -0.0108466|   0.95|
| cdf - none             |  -0.0266520|  -0.0441394|  -0.0120091|   0.95|
| dot50 - none           |  -0.0386676|  -0.0565360|  -0.0244286|   0.95|
| interval - text        |  -0.0142306|  -0.0395827|   0.0088330|   0.95|
| pdf - text             |  -0.0173831|  -0.0382481|   0.0010952|   0.95|
| pdfinterval - text     |  -0.0349320|  -0.0582309|  -0.0147643|   0.95|
| text60 - text          |  -0.0348015|  -0.0572445|  -0.0146841|   0.95|
| text99 - text          |  -0.0332327|  -0.0548334|  -0.0143890|   0.95|
| dot20 - text           |  -0.0467725|  -0.0696733|  -0.0270915|   0.95|
| cdf - text             |  -0.0472202|  -0.0697977|  -0.0287219|   0.95|
| dot50 - text           |  -0.0589688|  -0.0827837|  -0.0394403|   0.95|
| pdf - interval         |  -0.0026487|  -0.0224446|   0.0149646|   0.95|
| pdfinterval - interval |  -0.0205828|  -0.0417137|  -0.0020908|   0.95|
| text60 - interval      |  -0.0201490|  -0.0406122|  -0.0011737|   0.95|
| text99 - interval      |  -0.0186046|  -0.0393821|  -0.0011668|   0.95|
| dot20 - interval       |  -0.0319903|  -0.0521230|  -0.0144694|   0.95|
| cdf - interval         |  -0.0326434|  -0.0528762|  -0.0153672|   0.95|
| dot50 - interval       |  -0.0446749|  -0.0658727|  -0.0269324|   0.95|
| pdfinterval - pdf      |  -0.0179079|  -0.0320477|  -0.0040656|   0.95|
| text60 - pdf           |  -0.0173322|  -0.0314932|  -0.0033494|   0.95|
| text99 - pdf           |  -0.0159497|  -0.0286287|  -0.0037348|   0.95|
| dot20 - pdf            |  -0.0292403|  -0.0431578|  -0.0168155|   0.95|
| cdf - pdf              |  -0.0297922|  -0.0434379|  -0.0178445|   0.95|
| dot50 - pdf            |  -0.0418166|  -0.0567124|  -0.0293723|   0.95|
| text60 - pdfinterval   |   0.0007440|  -0.0143356|   0.0145822|   0.95|
| text99 - pdfinterval   |   0.0018328|  -0.0119685|   0.0154967|   0.95|
| dot20 - pdfinterval    |  -0.0113582|  -0.0253834|   0.0014423|   0.95|
| cdf - pdfinterval      |  -0.0119731|  -0.0258561|   0.0000023|   0.95|
| dot50 - pdfinterval    |  -0.0239050|  -0.0382035|  -0.0122357|   0.95|
| text99 - text60        |   0.0014208|  -0.0123420|   0.0145759|   0.95|
| dot20 - text60         |  -0.0118437|  -0.0259917|   0.0003234|   0.95|
| cdf - text60           |  -0.0123904|  -0.0265658|  -0.0010589|   0.95|
| dot50 - text60         |  -0.0244562|  -0.0385044|  -0.0130227|   0.95|
| dot20 - text99         |  -0.0132646|  -0.0258374|  -0.0016168|   0.95|
| cdf - text99           |  -0.0139052|  -0.0262479|  -0.0027021|   0.95|
| dot50 - text99         |  -0.0256314|  -0.0387142|  -0.0146853|   0.95|
| cdf - dot20            |  -0.0004947|  -0.0114182|   0.0098151|   0.95|
| dot50 - dot20          |  -0.0125241|  -0.0234830|  -0.0028291|   0.95|
| dot50 - cdf            |  -0.0119682|  -0.0220507|  -0.0026094|   0.95|
