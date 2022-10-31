

# ALZHEIMER PROGRESSION PREDICTION WITH NEUROIMAGING AND COGNITIVE TESTS;A MACHINE LEARNING MODELS STUDY
## Introduction: ## 
Alzheimer’s disease (AD)is the driving cause of dementia in older adults. Its frequency rate is expanding at a disturbing rate each year. A treatment given at an early stage of AD is more effective, and it causes less minor harm than a treatment done at a later stage. So, in this study we tried to train machine learning models for predict the dementia progression with just using numeric neuroimaging data and cognitive tests at baseline. In this training project we used AD datasets collected from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). The raw dataset (Merg, all collected data) were preprocessed used python and libraries like pandas in my notebook! we removed non-imaging features like blood biomarkers and etc. The first cleaned data imported and all columns heading converted to lower case for better manipulating. 
## Method: 
#### SUMMARY DATA DESCRIPTION: 
We have 11195 samples that include different participants(rid), with different visit session(visitcode). features data are demographics, numerical neuroimaging datas and cognitive tests. descriptive statistics are shown by describe() method.

#### STUDY DESCRIPTION:
We tried train a model(s) that can predict the progression of participants cognitive status using very non-invasive procedures (even blood biomarkers were exclude!). To aim this, we should first set the target. Our target for this study is the progression of participants dementia status from baseline. We just have diagnostic status for each visit session. CN for cognitive normal, MCI for mild cognitive impairment and Dementia for Alzheimer. We didn’t have target directly in data, so we needed to extract this from data by comparing the baseline diagnosis and last session diagnosis (at least 6 month later). We have different visit time session for participants and we considered the last one (from 6 to 186 month).
#### TARGET EXTRACTION SUDO CODE:
1-assign a numeric value to visit session(visitcode)(e.g. baseline = 1, month 6 = 2 and ...).
2-filter and split dataset by first visit (minimum value) and last visit (maximum value)
3-campare the diagnosis status (dx columne):
    if baseline is normal and still stay normal = low risk dementia progression (target = HCL)
    if baseline is normal and progress to MCI = risk of MCI (target = mMCI)
    if baseline is normal and progress to DEMENTIA = risk of AD (target = AD)
    if baseline is MCI and still stay MCI = stable MCI (target = sMCI)
    if baseline is MCI and progress to DEMENTIA = progressive MCI (target = pMCI)
    if baseline is MCI and last visit is normal! = data collection bias (remove data)
#### FEATURES SELECTION AND REDUCTION:
In this level we should select features. First, we used visualization and plots to select more correlated features. then we used a machine learning algorithm called Principal component analysis (PCA) to reduced features. The existence of many features and little data has the risk of overfitting. we used several methos to obtain the best results. 
Method 1 — Training the Model using all the Features: Before performing PCA on the dataset, some classifiers were used to train a model including all features in the dataset.
Method 2 — Training the Model using Reduced Features: For the next method, we examined the various features and tried to eliminate those features that had least correlated to the target. At the same time, we also should remove those features that exhibited multi-collinearity. The aim was to reduce the number of features and checked if the accuracy of the model can be improved.
Method 3 — Training the Model using Reduced Features (PCA): we applied PCA to the dataset in hope of model improving.  
Method 4 — Improving model use other methods: finally, we used KFOLD for a better data selection.
#### CONCLUSION: 
In this study we tried to train a Classifier Machine Learning Model to predict the dementia progression in Mild Cognitive Impairment patients and getting Alzheimer from healthy participants. We used just non-invasive features like medical imaging and cognitive test scores. After data cleanation and analysis, we traind two models, a Naive Bayes model and a Random Forest one. Then we tryed different technique to improve the models. Feature selection, feature reduction( principal component analysis(PCA)) and Kfold data extraction were used for this purpose. Finally its seems that Random Forest Classifier is the best model with about 85 accuracy.
