# advancedDatascience_CaseStudy
 
Case Description
Prof. Dr. Hendrik Meth
HdM, Stuttgart
Course Project Tasks & Grading

Element Description Share Due Date
Project Task
(team(s) of
four / five)

Analysis task with provided dataset:
Focus on preprocessing, structured
dataset. Sub Tasks
• Exploration
• Modeling
50% Upload presentation (and
implementation files) to Moodle
• Exploration: 12.6.2022 (Upload),
13.6.2022 (Presentation)
• Modeling: 19.6.2022 (Upload),
20.6.2022 (Presentation)

Oral Exam All topics covered in lecture and labs
part of the module, 15 minutes
50% 27.06.2022

# Dataset

Structured [text] data describing features of
houses and their price (label)
Method to predict label: Regression
Format:
–20.000 rows × 26 columns
–25 features
–1 label: price

Attribute Description 

1 id a notation for a house 
2 date Date house was sold 
3 price Price is prediction target (label) 
4 bedrooms Number of Bedrooms/House 
5 bathrooms Number of bathrooms/bedrooms 
6 sqft_living square footage of the home 
7 sqft_lot square footage of the lot 
8 floors Total floors (levels) in house 
9 waterfront House which has a view to a waterfront
10 dis_super Distance to next supermarket in feet
11 view Has been viewed
12 condition How good the condition is (Overall)
13 grade overall grade given to the housing unit
14 sqft_above square footage of house apart from basement
15 sqft_basement square footage of the basement
16 yr_built Built Year
17 yr_renovated Year when house was renovated
18 zipcode zip code
19 lat Latitude coordinate
20 long Longitude coordinate
21 sqft_living15 Living room area in 2015 (implies-- some renovations)
22 sqft_lot15 lotSize area in 2015 (implies-- some renovations)
23 ahf1 Additional house feature1
24 ahf2 Additional house feature2
25 ahf3 Additional house feature3


# Task 1: Exploration

Explore the provided data set using descriptive statistics (e.g. mean
values, standard deviations, min/max values, missing values) and
visualizations (e.g. histograms, boxplots)

Point out which data quality issues you identified in terms of
– Missing values
– Noise
– Outliers
– Features to be transformed, standardized or normalized

Point out which data reduction tasks would make sense from your
perspective in terms of
– Feature selection
– Instance selection

# Task 2: Modeling

Create a baseline regression model using the originally provided
dataset with minimal preprocessing and evaluate it based on
accuracies, using the training & validation subset of the data

Optimize your model by the following measures
– Data Preprocessing: Preprocess the original dataset to address the identified
issues applying different strategies for each issue
• data quality issues (Missing values, Noise, Outliers, Transformations)
• data reduction issues (Feature and Instance Selection)
– Algorithm Selection: Experiment with different regression algorithms, e.g. linear
regression, polynomial regression, regression trees etc.
– Hyper-parameter Tuning: Change the hyper-parameters of your algorithms (e.g.
„degree“ in case of polynomial regression)

# Task 2: Modeling continued
Implement a grid search to run and evaluate your different
model options

Your grid search should
– Receive a configuration JSON file as an input, including:
• Preprocessing steps to be conducted
• (Hyper-) Parameters of the preprocessing methods / algorithms
• Modeling algorithm to be applied
• Hyper-Parameters of the modeling algorithm
– Train and Evaluate models in an automated fashion, based
on the grid search configuration file
– Log all evaluation results (Validation and Test accuracies)
– Display an overview of all evaluation results in the end

# Task 2: Evaluation
Additionally to the 15.000 training and validation
examples provided, there will be 5.000 test examples,
which you will receive at the day of the presentation to
challenge your models and compare modeling results
between groups

Prepare to be ready to apply these test examples to
your best model – they will have a similar structure
like the training examples

# Task Presentation
20 mins presentation + 10 mins discussion
Structure your presentation based on the
bullet points of the task description
Every team member presents his/her part