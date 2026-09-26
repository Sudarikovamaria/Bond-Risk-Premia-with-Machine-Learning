# Bond Risk Premia with Machine Learning

**Contributors:** [Artemiy Maselskas](https://github.com/artemii2006) · [Maria Sudarikova](https://github.com/Sudarikovamaria) · [Anastasiya Polishchuk](https://github.com/AnastasiyaPolishchuk)


This project compares statistical and machine learning models for forecasting excess returns on Treasury bonds. Following Bianchi, Büchner, and Tamoni, the analysis examines whether different model classes extract useful predictive information from the Treasury yield curve and from a broad set of macroeconomic variables.

The paper motivating this project states that “machine learning methods, in particular extreme trees and neural networks (NNs), provide strong statistical evidence in favor of bond return predictability.” The project focuses on a systematic comparison of model performance across bond maturities and information sets, including yield-only and macro-augmented specifications.


## **Research Problem**

Bond risk premia vary over time and may depend on nonlinear relationships between interest rates, macroeconomic conditions, and financial variables. Traditional linear models may therefore fail to capture all relevant information contained in the data.

The main research problem is to determine whether machine learning models provide more accurate out-of-sample forecasts of bond excess returns than traditional approaches. A related question is whether macroeconomic variables improve forecasting performance relative to models based only on the yield curve.


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

Excess returns are expressed in percent throughout. This convention has to be shared across model families rather than treated as a formatting detail: Elastic Net is not invariant to the scale of the target, because rescaling the target changes its quadratic loss and its ridge penalty quadratically while changing its lasso penalty only linearly, and no single penalty parameter reconciles the two. On this sample, switching between percent and decimal units moves the Elastic Net out-of-sample $R^2$ by up to 3.8 percentage points. Ridge and Lasso are invariant.


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


## **Models Specifications**

### **Linear and Factor-Based Models**

This group covers Panels A and B of both tables in the paper and is implemented in `models/PCA_PLS_linear_models/`.

**Yield-only specifications (Table 1).** Principal component regressions on the first three, five, and ten components of the forward-rate cross-section; the same regressions augmented with the squared components; partial least squares on three and five components; and Ridge, Lasso, and Elastic Net on the ten forward rates directly. With ten components the principal component regression is, up to a rotation of the basis, the original Cochrane–Piazzesi specification.

**Macro-augmented specifications (Table 2).** A regression on the first eight principal components of the macroeconomic panel together with the Cochrane–Piazzesi factor; the subset specification of Ludvigson and Ng (2009), $F_t = (F_{1t}, F_{1t}^3, F_{3t}, F_{4t}, F_{8t})$, with the same factor; partial least squares on eight components; and the three penalized regressions in two variants — one using the Cochrane–Piazzesi factor as a single additional regressor, the other using all ten forward rates.

**Estimation details.** The Cochrane–Piazzesi factor is the fitted value of a regression of the average excess return on all forward rates, and the macroeconomic principal components are extracted by principal component analysis; both are re-estimated inside every expanding window, so neither introduces information unavailable at the forecast date. Predictors are standardised using training-sample moments only. Penalties for Ridge are selected from a fixed logarithmic grid on the validation sample; for Lasso and Elastic Net the penalty is selected along the regularisation path, which starts at the penalty that sets every coefficient to zero and descends by three orders of magnitude. An absolute grid is inappropriate here because the relevant scale depends on the dispersion of the target, and the ten-year excess return is roughly six times as volatile as the two-year one.

**Forecast timing.** The first forecast is made in January 1989 and is realised in January 1990, which yields 348 out-of-sample observations. The statement in the paper that the recursive forecast starts in January 1990 refers to the realisation date: Section 4.1 specifies that the first forecast error compares the excess return over February 1989 to January 1990 with the forecast made in January 1989.

A twelve-month holding-period return dated $t$ is only observed at $t+12$, so training on targets dated up to $t-1$ uses information unavailable at $t$. The `GAP` constant controls this: `GAP = 1` reproduces the design of the paper, while `GAP = 12` removes the look-ahead. Both variants are reported, because the difference is large and the published corrigendum to the paper revises the original results for exactly this reason.

### **Tree-Based Models**

This group includes Gradient Boosted Regression Trees, Random Forest, and Extra Trees models. The models are estimated separately for each target maturity and information set.

### **Neural Network Models**

This group includes feed-forward neural networks with different numbers of layers and hidden nodes, as well as group-ensemble specifications based on forward-rate and macroeconomic information.


## **Results**



## **References**

1. Bianchi, D., Büchner, M., & Tamoni, A. (2021). *Bond Risk Premia with Machine Learning*. Author’s Accepted Manuscript, University of Warwick Research Archive. [WRAP version](https://wrap.warwick.ac.uk/151797/1/WRAP-bond-risk-premiums-machine-learning-B%C3%BCchner-2021.pdf).

2. Bianchi, D., Büchner, M., & Tamoni, A. (2021). *Corrigendum: Bond Risk Premiums with Machine Learning*. The Review of Financial Studies, 34(2), 1090–1103. The authors revise the published results after correcting for the use of information unavailable at the forecast date, and report lower out-of-sample $R^2$. The freely available manuscripts predate this correction, so their tables are the pre-correction ones. [DOI](https://doi.org/10.1093/rfs/hhaa098).

3. Liu, Y., & Wu, J. C. (2021). *Liu–Wu Yield Data* [Data set]. Monthly and daily yield curve data. [Official data page](https://sites.google.com/view/jingcynthiawu/yield-data).

4. Federal Reserve Bank of St. Louis. (2018). *FRED-MD: December 2018 Vintage* [Data set]. [Official FRED-MD data page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases).
