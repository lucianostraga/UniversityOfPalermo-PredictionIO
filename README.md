# Multinomial Naive Bayes PIO Engine (7 features implementation) for predicting drop-out rate at University of Palermo (Buenos Aires, Argentina)

## This engine supports training sets of seven features, and bulk queries as an array of JSONs with the attributes.

Also includes, python scripts for importing data, evaluating accuracy by PCA and t-SNE training set transformations, and training set plotting. Python scripts are based on Scikit Learn machine learning library.

## FULL PDF IEEE PAPER - "Machine Learning with Salesforce and Apache PredictionIO (incubating) in the Academic World" by Luciano Straga [PDF Paper]https://github.com/lucianostraga/UniversityOfPalermo-PredictionIO/blob/master/IEEE_Paper.pdf

Decision Boundary after PCA transformation - Gaussian Naive Bayes

![PCADecisionBoundary](https://github.com/lucianostraga/UniversityOfPalermo-PredictionIO/blob/master/images/decisionPCA.png)

Training set plotted after PCA transformation (7 variables -> 2 variables (x,y) )

![PCAset](https://github.com/lucianostraga/UniversityOfPalermo-PredictionIO/blob/master/images/TraningDataPCA.png)

Training set plotted after t-SNE transformation (7 variables -> 2 variables (x,y) )

![t-SNEset](https://github.com/lucianostraga/UniversityOfPalermo-PredictionIO/blob/master/images/TrainingDataTSNE.png)


