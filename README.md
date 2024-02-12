# A full Data Science and Mlops project

## Complete Data analysis and **CI/CD** Pipeline for clustering and classification model of Telco Churn dataset

## Motivation
The motivation behind this project is to develop a complete data analysis of the Telco Churn dataset, looking for **key patterns** that allows the creation of a clustering model to **stablish different categories of clients**, obtain the **main features behind those clients** and finally obtaining a classification model to predict whether the client will Churn or not.

## Dataset
The dataset is obtained from the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) Kaggle dataset. This dateset includes information about: 
- Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## Main findings
### Clustering model
From the clustering analysis, the main findings correspond to the main characteristics of **3 groups of clients**, where the behaviour of clusters is summarized using **SHAP values**. From the results it is observable that the **5 most important features** in the clustering model are:
- **TotalCharges**: The total charges amount that clients have paid.
- **MonthlyCharges**: The amount of charges a client pays monthly. 
- **tenure**: The amount of time a client have been using the company services.
- **InternetService_No**: Amount of clients that do not have internet service.
- **Partner**: Binary value to determine whether a client have a partner or not.



![image file](/img/eda_results.png)
### Classification model
From the classification analysis, the main findings are:
- **A higher tenure time** usually leads to a **lower Churn rate**.
- People with **Month to Month contracts have a higher Churn rate**.
- **Monthly charges have a high influence in Churn rate**.
- Telco **fiber optic internet service could be bad** or Telco **have external competition at lower prices that offers cheap fiber optic**.
- Clients with **no online security** or  **no tech support** have a **higher Churn rate**.
- Clients with a **two-year contract** have a **lower Churn rate**. 



![image file](/img/churn_results.png)


After creating the model, **SHAP explainer** is also saved to generate a `shap.plots.waterfall()` for different clients in a `streamlit_app`.


## How are the models and experiments tracked?
Models and experiments are tracked using `Mlflow` library linked to a remote repository in **dagshub** where data and models metadata is saved.


## Repository organization
The main components of this repository are:
- [EDA and Clustering notebook](research/EDA_churn_analysis.ipynb), where the complete EDA and Clustering model using ``K-Prototypes`` library is presented. 
- [Churn model development notebook](research/Model_churn_analysis.ipynb), where different classification models are tested and evaluated, with an ``XGBoost`` model being the better fit.
- [Components of the models made .py files](src/Churn_analysis/components), where main components of the analysis are productionalize into ``.py`` files.
- [DVC pipeline](dvc.yaml). The ``.yaml`` file that controls the `dvc repro` actions.
- [Workflow pipeline](.github/workflows/dvc_pipeline.yaml)
The ``.yaml`` file that controls the `github actions`.
- [API development](app.py)
The ``.py`` where a `FastAPI` is presented.

**The main workflow of the code presented in this reposity follows this scheme:**
![image file](/img/churn_repo.jpg)

## How to run the pipeline?

Start by installing the requirements.txt

```python
pip install -r requirements.txt
```
Then start `dvc`
```python
dvc init
```
And proceed to connect to the remote storage using your selected login system
```python
dvc remote add MyremoteStorage
```

Then define `Mlflow` credentials in `src/Churn_analysis/components/model_evaluation.py` file or define it as `env variables` in github:
```python
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/User/Repo.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="MyUserName"
os.environ["MLFLOW_TRACKING_PASSWORD"]="MyPassword"
```
Then run the pipeline
```python
dvc repro
```
> [!CAUTION]
> This would just reproduce the pipeline, generate the model, and log the info to mlflow.