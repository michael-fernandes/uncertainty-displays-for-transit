# Data and code for *Uncertainty Displays Using Quantile Dotplots or CDFs Improve Transit Decision-Making* (CHI 2017)

[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3173574.3173718-blue.svg)](http://doi.org/10.1145/3173574.3173718)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.1136329-blue.svg)](https://doi.org/10.5281/zenodo.1136329)

*Michael Fernandes ([mfern@uw.edu](mailto:mfern@uw.edu))*<br>
*Logan Walls ([logan.w.gm@gmail.com](mailto:logan.w.gm@gmail.com))*<br>
*Sean Munson ([smunson@uw.edu](mailto:smunson@uw.edu))*<br>
*Jessica Hullman ([jhullman@uw.edu](mailto:jhullman@uw.edu))*<br>
*Matthew Kay ([mjskay@umich.edu](mailto:mjskay@umich.edu))*

This repository contains data and analysis code for the following paper:

Michael Fernandes, Logan Walls, Sean Munson, Jessica Hullman, and Matthew Kay. "Uncertainty Displays Using Quantile Dotplots or CDFs Improve Transit Decision-Making", *Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems - CHI 2018*. DOI: [10.1145/3173574.3173718](http://doi.org/10.1145/3173574.3173718)


## Final analysis

The final analysis is outlined in [final_analysis.md](final_analysis.md) (also available in [html](final_analysis.html)). This analysis was compiled from an RMarkdown notebook, [final_analysis.Rmd](final_analysis.Rmd).

Additional materials from the final analysis are available here:

* [data/final_trials.csv](data/final_trials.csv): data from the final study

* [models/final_model.rds](data/final_trials.csv): fitted model object from final Beta regression. This is a `brmsfit` R object.

* [experiment_screen_shots.pdf](experiment_screen_shots.pdf): Screenshots of the major end points encountered by subjects in our online experiment. Includes an example tutorial that walks a subject through the important details of how to use one of the uncertaintiy visualizations (probability density plots).


## Pre-registration

Our pre-registered analysis plan is in [pre-registration.pdf](pre-registration.pdf). It is also available for verification on [AsPredicted](https://aspredicted.org/iv7jb.pdf).


## Pilot analyses

Some (rougher) pilot analysis notebooks are also included in this repository:

* [valuation_analysis.pdf](valuation_analysis.pdf): Valuation analysis that helped us set payoffs.

* [pilot_exploration.ipynb](pilot_exploration.ipynb) / [pilot_exploration.html](pilot_exploration.html): An iPython notebook with some preliminary analysis of pilot data.
