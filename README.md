# HyIPO: Hyped Initial Product Offerings

## Project Intro/Objective
The purpose of this project is to leverage statistical methods to study risk in the IPO market and to build a model to rank IPOs in terms of their expected returns.

### Methods Used
Inferential Statistics, Feature engineering, Machine Learning, Deep Learning, Data Visualization, Predictive Modeling

### Technologies
Jupyter notebook, Python 3.7.13 (Scikit-learn, Pandas, Numpy, Tensorflow, Requests_html, Matplotlib and Seaborn), PyCaret

## Project Description
The stock exchange is an intriguing and fascinating world where bulls and bears constantly fight for the leading position, where fundamental market forces are best displayed, and operators’ psychological traits and biases drive prices and valuations. 
Each year private companies looking for liquidity undergo the IPO process thus opening their doors to both institutional and private investors. These newcomers to the stock arena are characterized by a relatively large amount of uncertainty compared to already listed entities as the transition to becoming a publicly traded company typically involves numerous changes in terms of transparency, ownership structure and corporate governance.    

When building a model to rank IPOs in terms of their expected returns, I had to overcome two main challenges:
- Investors base an average of 40% of their IPO investment decisions on non-financial factors, especially quality of management, corporate strategy and execution, brand strength and operational effectiveness, a compelling equity story and corporate governance (according to an EY report); all these factors are difficult to be quantified properly by a machine learning model.
- The success or failure of an IPO is greatly determined by an accurate pricing. Valuation and pricing are complex processes and are hugely important components of the IPO process. Given the fact that often there are divergent interests of the issuer and the underwriter, the valuation and pricing do not necessarily go hand in hand. A conclusion quantifying underwriters' conflict of interest cannot be drawn via a predictive model.
 
### Data acquisition and understanding 
- data sources and formats 
- features extraction
- cleaning and wrangling 
 
The approach consists in using domain knowledge and extracting relevant features to improve the quality of results from the machine learning process.
My target variable is "label 1" if the 1st-day change in price after the IPO date is higher than the risk-free rate benchmark (5-year Treasury bill rate) and "label 0" otherwise.

The first step is to find a reliable source of historical data regarding IPOs. I used the dataset provided by the IPOScoop website (https://www.iposcoop.com/scoop-track-record-from-2000-to-present/) which includes information about the Issuer, Symbol, Rating, IPO date, IPO price, the 1st-day returns and the IPO managers for the period 2000-2020 ("the IPOs list" - 3633 observations). 

For each issuer in the dataset, I executed the following steps:
- data cleaning: standard procedures (changing data types, checking whether NaN's and/or Null values exist, dropping useless columns, etc.) plus checking and replacing the issuers' trading symbols in the IPO list with the accurate ones from a dataset which captures the US Publicly Listed Companies as of today (https://stockanalysis.com/) 
- feature engineering and extraction:
  - the 1st-week and 1st-month closing prices subsequent to the IPO (Yahoo Finance) for each issuer in the IPOs list 
  - market performance indicator: the S&P500 change in closing prices for 1 week, 1 month and 3 months prior to the IPO date (Yahoo Finance)
  - market volatility indicator: the VIX change in closing prices for 1 week, 1 month and 3 months prior to the IPO date (Yahoo Finance)
  - AAII Investor Sentiment Survey (https://www.aaii.com/sentimentsurvey/sent_results) published during the week prior to the IPO date 
  - label 1 if the Lead/Joint-Lead Managers are Tier 1 underwriters or 0 otherwise
  - date-based features (day of the week)
  - social indicator: search data from google trends API (https://trends.google.com/) in order to assert potential investors’ appetite for each IPO; I used the number of spikes in popularity (if observation > mean) during the last 2 weeks prior to the IPO 
  - 5-year Treasury bill historical rate for each IPO date

### Data analysis and features importance
The feature selection process of finding and selecting the most useful ones in a dataset is done by combining correlation analysis with applying ML models with different sets of features and then examining feature importance techniques and the metrics. Unnecessary or correlated features decrease training speed, model interpretability and the generalization performance on the test set. I settled upon the following set of features: 'Star Ratings',  'S&P 3 Months % Px Chng', 'VIX 1 Week % Px Chng', 'Sentiment survey', 'Tier1 IB'. 

### Models – binary classification problem
- split data into train/ test sets
- define a pipeline for preprocessing
- use PyCaret to determine the 5 best model in terms of accuracy
- fine-tune the hyperparameters of the best models using Grid Search
- evaluate the performance of the models on the test dataset
- build NN architectures and experiment with more aspects of Dense NN models such as layer activations, learning rates, regularization
- select the best model in terms of accuracy, f1-score and explainability – Logistic Regression

### Conclusions
According to the coefficients resulted from my selected model (Logistic Regression) I can draw the following conclusions:
- the rating (1 to 5 hierarchical values) of the IPO has a positive impact on the odds that the IPO returns represented in the observation are in the target class (“1”) (a Star Rating (Wall Street Consensus of Opening-day Premiums), is a consensus taken, at press time, from Wall Street and investment professionals concerning how well an IPO might perform when it starts trading)
- the market performance indicator - the change in S&P500 closing prices for 3 months prior to the IPO date has a positive impact on the odds that the IPO returns represented in the observation are in the target class (“1”) 
- to forward-looking AAII Investor Sentiment Survey (bull-bear spread), published during the week prior the IPO date has a positive impact on the odds that the IPO returns represented in the observation are in the target class (“1”) (the participants in the survey answer the following question: what direction do AAII members feel the stock market will be in the next 6 months?)
- the Lead/Joint-Lead Managers being Tier 1 underwriters date has a slightly positive impact on the odds that the IPO returns represented in the observation are in the target class (“1”) 
- the market volatility indicator - the change in VIX closing values for 1 week has a positive impact on the odds that the IPO returns represented in the observation are NOT in the target class (“1”) (the CBOE Volatility Index, or VIX, is a real-time market index representing the market's expectations for volatility over the coming 30 days)

### Limitations and future recommendations
- after conducting web scrapping with the requests HTML package and retrieving the URLs for the S-1 and F-1 reports from SEC website, some of the reports were missing and some were incorrectly selected; future recommendations are: improve the web scrapping technique to properly parse the S-1 Reports and perform a sentiment analysis, extract financial factors, particularly debt to equity ratios, EPS growth, sales growth, ROE, profitability and EBITDA growth and examine the risk factors section of the reports
- when analyzing social indicators - explore data from google trends API - in order to assert potential investors’ appetite for each IPO, more specifically the number of spikes in popularity during the last 2 weeks before the IPO, I cannot state if the spikes are due to good or bad news; future recommendations: develop a framework for automatically distilling stock market insights from an online conversation (news articles, social media posts, etc.) before the IPO date and perform a further sentiment analysis
- future recommendations: build a benchmark with competitors and comparable companies for each issuer in order to assess the analyzed company’s business fundamentals with its peers group
