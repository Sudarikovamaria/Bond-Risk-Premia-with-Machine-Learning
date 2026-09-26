# Bond Risk Premia with Machine Learning

**Contributors:** [Artemiy Maselskas](https://github.com/artemii2006) · [Maria Sudarikova](https://github.com/Sudarikovamaria) · [Anastasiya Polishchuk](https://github.com/AnastasiyaPolishchuk)

---

This project compares statistical and machine learning models for forecasting excess returns on Treasury bonds. Following Bianchi, Büchner, and Tamoni, the analysis examines whether different model classes extract useful predictive information from the Treasury yield curve and from a broad set of macroeconomic variables.

The paper motivating this project states that “machine learning methods, in particular extreme trees and neural networks (NNs), provide strong statistical evidence in favor of bond return predictability.” The project focuses on a systematic comparison of model performance across bond maturities and information sets, including yield-only and macro-augmented specifications.

---

## **Research Problem**

Bond risk premia vary over time and may depend on nonlinear relationships between interest rates, macroeconomic conditions, and financial variables. Traditional linear models may therefore fail to capture all relevant information contained in the data.

The main research problem is to determine whether machine learning models provide more accurate out-of-sample forecasts of bond excess returns than traditional approaches. A related question is whether macroeconomic variables improve forecasting performance relative to models based only on the yield curve.

---

## **Data**

### **Bond Yield Data**

The project uses monthly Treasury yield data based on the Liu–Wu yield curve dataset. The sample covers the period from August 1971 to December 2018 and includes maturities used to construct forward rates and bond excess returns.

### **Macroeconomic Data**

Macroeconomic predictors are obtained from the FRED-MD database. The variables are transformed according to the transformation codes provided with the dataset and aligned with the bond yield data at a monthly frequency.

### **Raw Data**

The raw-data inputs are:

- Liu–Wu monthly Treasury yields;
- the December 2018 vintage of FRED-MD.

### **Processed Data**

The processed datasets include:

- `forward_rates.csv` — the short rate and forward rates from two-year to ten-year maturities;
- `excess_returns.csv` — twelve-month excess returns for two-, three-, four-, five-, seven-, and ten-year bonds;
- `macro_panel.csv` — transformed FRED-MD macroeconomic variables;
- `fred_md.csv` — the aligned original FRED-MD variables;
- `yield_only_data.csv` — yield-based predictors combined with the target variables;
- `macro_data.csv` — yield-based and macroeconomic predictors combined with the target variables.

---

## **Methodology**

### **Target Variables**

The target variables are twelve-month excess returns for Treasury bonds with maturities of 2, 3, 4, 5, 7, and 10 years.

### **Predictor Variables**

The predictor sets consist of:

- the short rate and forward rates derived from the Treasury yield curve;
- macroeconomic variables from FRED-MD;
- factor representations and selected transformations used by the corresponding model specifications.

### **Forecasting Design**

Following the forecasting design in the paper, the data are divided into three chronological subsamples: a training set, a validation set, and a testing set. The training set contains the first 85% of the in-sample observations, while the validation set contains the remaining 15%. The validation set is used to select hyperparameters by minimizing the validation forecasting error. Randomly selecting observations is avoided in order to preserve the time-series structure of the data.

The testing set is the out-of-sample period and begins in January 1990. Forecasts are produced recursively using an expanding window: after each forecast date, one additional monthly observation is added to the in-sample period, while the 85%/15% training-validation proportion is maintained. The selected model is then refitted using the available in-sample history before generating the next one-year-ahead excess-return forecast. For PCA and standard linear regressions, the paper does not require a separate validation stage, so the pre-testing observations are treated as one in-sample period.

### **Evaluation Metrics**

Model performance is evaluated using:

- mean squared forecast error;
- benchmark mean squared error;
- out-of-sample $R^2$ relative to the historical-mean benchmark.

---

## **Models Specifications**

### **Linear and Factor-Based Models**

This group includes principal component analysis, partial least squares, and penalized linear regressions. The penalized regressions include Ridge, Lasso, and Elastic Net specifications based on either the Cochrane–Piazzesi factor or forward rates.

### **Tree-Based Models**

This group includes Gradient Boosted Regression Trees, Random Forest, and Extra Trees models. The models are estimated separately for each target maturity and information set.

### **Neural Network Models**

This group includes feed-forward neural networks with different numbers of layers and hidden nodes, as well as group-ensemble specifications based on forward-rate and macroeconomic information.

---

## **Results**


---

## **References**

1. Bianchi, D., Büchner, M., & Tamoni, A. (2021). *Bond Risk Premia with Machine Learning*. Author’s Accepted Manuscript, University of Warwick Research Archive. [WRAP version](https://wrap.warwick.ac.uk/151797/1/WRAP-bond-risk-premiums-machine-learning-B%C3%BCchner-2021.pdf).

2. Liu, Y., & Wu, J. C. (2020). *Reconstructing the Yield Curve*. NBER Working Paper No. 27266. [NBER version](https://www.nber.org/papers/w27266).

3. McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research. *Journal of Business & Economic Statistics*, 34(4), 574–589. [DOI](https://doi.org/10.1080/07350015.2015.1086655).
