# Model Card

A classification model was developed on publicly available [Census Bureau data](https://archive.ics.uci.edu/ml/datasets/census+income) and code was used to monitor the model performance on various data slices with the help of FastAPI package.

## Model Details

Model used was a Random Forest Classifier with the Scikit-learn framework.

## Intended Use

Primary use of the data is to predict whether income more or less of $50K/yr are based on various features. Used for income prediction and understanding certain features and impact towards income.

## Training Data

The model was trained on 80% of the [Census Bureau data](https://archive.ics.uci.edu/ml/datasets/census+income) and remaining 20% used for the testing model. The data set contained 32561 data samples with the following columns: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss,hours-per-week, native-country.

## Evaluation Data

The evaluation data consisted of 20% of the Census Bureau data. 

## Metrics

The primary metrics were Precision: 0.7315, Recall: 0.6382, and F1-score: 0.6816. 

## Ethical Considerations

The data contains some features that could hold some bias based around race and gender which the data was trained on. Data should be interpretted with the results with that bias in mind. 

The data was public but any data inputted into the model should be considered for review with any privacy regulations. 

## Caveats and Recommendations

The model was used only for testing and research purposes and not for any production use. 

The model will only perform based around the data it was provided meaning bad data in, equals bad data out. The Census data provided for the model may not be the best predictor for the general population. Better data sets and large sample size would improve the model.