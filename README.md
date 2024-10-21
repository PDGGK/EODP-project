# Crime Rate Prediction in Victoria

## Project Overview
This project aims to explore the impact of various socio-economic factors on the crime rates in different suburbs of Victoria, Australia. We specifically analyze how government manipulation of these factors can influence crime rates, helping guide policy-making towards a safer and more livable community. Using both linear regression and decision tree regression models, we predict the suburb-level crime rate based on social, economic, and regional features.

### Group Members
- Sheldon Liu (1455927)
- Zihan Dai (1488802)
- Jiahao Ni (1455828)
- Phone Moe Thway (1342981)

## Research Question
How do different socio-economic factors influence crime rates in communities across Victoria?

### Motivation
Insights from this research can assist the government in effectively reducing crime rates and improving quality of life, ultimately helping Melbourne maintain its reputation as the world's most livable city.

## Datasets Used
1. **Victorian Communities Dataset**: 452 distinct suburbs and 34 LGAs
2. **House Price Dataset**: Information for 774 distinct suburbs from 2013 to 2023
3. **Electronic Gaming Machine (EGM) Dataset**: Records for 57 LGAs from 2011 to 2020
4. **LGA Offences Dataset**: Offence data for 79 LGAs, detailing different types of offences and their locations

## Methodology
### 1. Data Preprocessing
Data preprocessing included handling null values, formatting columns consistently, and merging datasets based on common locality and LGA identifiers. Specific preprocessing actions:
- Removal of rows with missing data.
- Standardization of suburb and LGA names.
- Imputation of missing values using column means.
- Removal of redundant or irrelevant columns to streamline the dataset. Specifically, columns such as `Year Ending`, `Police Region`, and redundant identifiers like `Locality Description` were removed as they were not relevant to the analysis.

### 2. Feature Engineering and Selection
- **Feature Engineering**: We calculated average house prices, average EGM losses, and new features such as the number of hospitals and aged care facilities.
- **Feature Selection**: Based on correlation analysis and multicollinearity checks, key features were selected to avoid redundancy and improve model accuracy.

### 3. Regression Modeling
We implemented two different regression models:
1. **Linear Regression**: Used as a baseline to assess linear relationships between features and target variables.
2. **Decision Tree Regression**: Applied to capture potential non-linear relationships, providing a more flexible model structure.

### 4. Model Evaluation
We evaluated both models using several metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-Squared (R² Score)**

Linear regression was found to perform better overall, with an R² score of 0.6414 compared to 0.3595 for the decision tree model. The decision tree model struggled with hyperparameter tuning due to limited time for training and optimization.

## Results and Key Findings
### Exploratory Data Analysis
- **Melbourne Focus**: We found that Melbourne was the most representative region for crime data due to its high crime rates, reflecting broader patterns that could help develop region-specific policies.
- **Trends Observed**: EGM losses and house prices both exhibited significant growth from 2014 to 2016, correlating with a rise in property crimes. After 2017, a decline was observed, possibly due to new regulations on EGM machines.

### Correlation Analysis
- **Economic Factors**: Unemployment rate, income level, and public housing rates showed significant positive correlations with crime rate.
- **Housing Prices**: Correlations between average house price and crime rate were low, suggesting a less direct impact.
- **Aboriginal Population and Crime**: High correlation found between higher Aboriginal population percentages and crime rates, potentially pointing to socio-economic disparities and challenges.

### Model Performance
The linear regression model performed better due to the mostly linear relationships among features. The decision tree model could benefit from further hyperparameter tuning.

## Limitations and Potential Improvements
1. **Dataset Recency**: Some datasets had only values for 2014, limiting temporal analysis.
2. **Outliers**: Average values over the years may be skewed by outliers; improvements could include filtering out extreme values for a more reliable average.
3. **Future Models**: Incorporate time-series models to better understand trends over time.
4. **Specificity of Features**: Some features could be better defined, such as investigation statuses, for more meaningful analysis.

## Conclusion
This project identified several socio-economic, regional, and community-specific factors that influence crime rates across Victoria. By developing predictive models, we provided the basis for government policy-making to address these influencing factors more effectively.

While the regression models give insight into how changes in socio-economic conditions could impact crime, further research, including finer-grained models and more recent datasets, could yield even more reliable recommendations for crime prevention.

## Usage
1. **Clone Repository**: `git clone <repository_url>`
2. **Install Dependencies**: Install required Python libraries using `pip install -r requirements.txt`.
3. **Run Notebooks**: The Jupyter notebooks for data processing and modeling are available in the `/notebooks` directory. Start with `code.ipynb`.

## References
- Sydney Morning Herald, Five Reasons Why Melbourne is No Longer the World's Most Livable City (2017).
- Australian Institute of Family Studies, "Gambling in Suburban Australia."
- Crime Statistics Agency Victoria: Latest Crime Data by Area.
- Australian Institute of Criminology.

For more details, please see the report in `/docs` and the accompanying presentation slides.
