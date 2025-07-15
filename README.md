# **Time series model predicting the domestic consumption of almonds,shelled basis agricultural crop**

This project's goal is to undertand historical trends and build simple predictive model for domestic almonds consumption while performing an explaratory data and time series analysis.
I used a psd_alldata dataset, it can be found on kaggle, its kind of too big saving here.

# The key steps in the building of the model include;
data loading and initial inspection/, 
handling missing data,
filtering for almonds shelled basis and relevant attribute, 
Time series preparation(creating datatime index, handling missing periods)
then moved on to stationary testing using ADF test/
then differencing to achieve stationarity/
feature Engineering/ 
Data splitting/
model building, a linear regression model/
then model evaluation.

# The KEY FINDINGS were increasing production,and fluctuations in imports or exports 
some of the evaluation metrics like MAE (MEAN ABSOLUTE ERROR) was off with about 1718.85

ROOT MEAN SQUARED ERROR was whcich inidcataed a typical magnitude of prediction errors
R-SQUARED is 0.976, suggesting 97.6% of variance in the test set, this model may be performing well by chance, boosting consumption of almonds can be done through more market advertising, market expansion, pricing strategies, supply chain efficiency, boosting final value of almonds, improving trade policies

**please note that this wasn't analysed at a granular level and factors influencing demand and supply can vary significantly by region and time of year. this model can further be improved in the future.**
