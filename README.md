# Data and code for *Better Deciding through Discretizing: Uncertainty Displays to Improve Transit Decision-Making* (2017)

## Repository contents

* [pilot_exploration.ipynb](pilot_exploration.ipynb)[pilot_exploration.html](pilot_exploration.html): An iPython notebook with some preliminary analysis of pilot data.
* [pre-registration.pdf](pre-registration.pdf): Our pre-registered analysis plan. Also available for verification on [AsPredicted](http://aspredicted.org/blind.php?x=g2yb2f).
* [final_analysis.html](final_analysis.html)/[.md](final_analysis.md)/[.Rmd](final_analysis.Rmd): RMarkdown analysis of the final study and results.
* [experiment_screen_shots.pdf](experiment_screen_shots.pdf): Screenshots of the major end points encountered by subjects in our online experiment. Includes an example tutorial that walks a subject through the important details of how to use one of the uncertaintiy visualizations (probability density plots).
* [data/](data/): Data used in the analyses
    * [data/final_trials.csv](data/final_trials.csv): data from the final study
* [models/](models/): Fitted Bayesian models
    * [models/final_model.rds](data/final_trials.csv): fitted model object from final Beta regression. This is a `brmsfit` R object.
