# Credit-Risk-Probability-Model-for-Alternative-Data

## Overview

The **Credit Risk Probability Model for Alternative Data** is an advanced data science project designed to assess credit risk using non-traditional (alternative) data sources. This repository demonstrates the end-to-end process of developing, evaluating, and deploying a machine learning model to predict the probability of credit default, with a particular emphasis on the utilisation of alternative data attributes. The project is crafted for data scientists, risk analysts, and financial technologists interested in modern credit risk assessment methodologies.

## Key Features
- **Behavioural Data Integration:** Utilises alternative, non-traditional data sources to capture a broader spectrum of financial behaviours.
- **Exploratory Data Analysis (EDA):** Visualisations and statistical summaries to understand data distributions and relationships.
- **Comprehensive Data Preprocessing:** Handles missing values, outliers, feature engineering, and encoding of categorical variables.
- **Feature Engineering:** Methods for constructing predictive features from behavioural data, including Weight of Evidence (WoE) encoding.
- **Predictive Model Development and Interpretability:** Focuses on explainable models (e.g., Logistic Regression, Random Forest, and Gradient Boosting with WoE encoding) to meet regulatory requirements (Basel II).
- **Model Evaluation:** Evaluates models using metrics like ROC-AUC, Precision-Recall, F1-score, and calibration plots.
- **Deployment-Ready:** Includes scripts for model serialisation and API deployment (via FastAPI).
- **Data Versioning:** Uses DVC for reproducible data pipelines.
- **Experiment Tracking:** Supports MLflow or similar tools for model tracking.
- **Automated Testing:** Includes test scripts and CI/CD setup.
- **Jupyter Notebook Implementation:** The codebase is primarily in Jupyter Notebook format, facilitating transparency, reproducibility, and ease of experimentation.
- **Customisable Workflows:** Designed for adaptability, allowing users to tailor feature engineering and model parameters to their specific data and requirements.

---

## Table of Contents

- [Project Background](#project-background)
- [Credit Scoring and Business Understanding](#credit-scoring-and-business-understanding)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Interpretation](#results-and-interpretation)
- [Testing](#testing)
- [Contribution](#contribution)
- [Project Status](#project-status)
- [Acknowledgements](#acknowledgements)

## Project Background

Traditional credit risk models often rely on limited financial data such as credit bureau reports, income statements, and payment histories. However, many potential borrowers, especially in emerging markets, lack extensive credit histories. To address this gap, alternative data such as mobile phone usage, utility payments, social media behaviour, and other digital footprints can be leveraged to enhance credit risk assessment.

This project explores the use of shopping behaviour, transaction patterns, and other behavioural indicators as alternative data sources. It leverages these insights to build a robust probability-of-default model, promoting financial inclusion and improving risk management for lenders.

## Credit Scoring and Business Understanding

### 1. Why Model Interpretability Matters under Basel II
The Basel II Accord requires financial institutions to rigorously measure, monitor, and manage credit risk, especially for capital adequacy calculations. This regulatory framework places a premium on transparency, interpretability, and auditability of risk models. As a result, credit scoring models must be not only accurate but also interpretable and well-documented. This ensures regulators, auditors, and internal stakeholders can clearly understand how scores are generated, how input variables affect outcomes, and how risk is being quantified, thus maintaining compliance and facilitating trust in the model’s outputs.

### 2. The Need for a Proxy Default Label
In the absence of a direct "default" label in the data, one must define a proxy variable (such as "90+ days past due") to approximate default behavior. This proxy is essential for supervised learning, as models require a clear target variable. However, predictions based on a proxy carry business risks: they may not fully capture the real-world definition of default as used by regulators or the institution, leading to potential misclassification. This can result in inaccurate risk assessments, suboptimal lending decisions, or regulatory misalignment if the proxy diverges significantly from actual default patterns.

### 3. Key Trade-offs: Simplicity vs. Complexity in Model Choice
Simple, interpretable models (e.g., Logistic Regression with Weight of Evidence encoding) offer clear explanations for decisions, easier validation, and regulatory acceptance—crucial for compliance and governance. However, they may sacrifice some predictive power compared to more complex approaches. High-performance models (e.g., Gradient Boosting Machines) can yield better accuracy but are harder to interpret and validate, increasing the risk of regulatory challenges and implementation hurdles. In regulated environments, the trade-off often favors interpretability and transparency over marginal gains in predictive accuracy, unless the complexity can be justified and thoroughly documented.


## Data Sources

- **Alternative Data:** Anonymised shopping behaviour signals, customer identity and behavioural segmentation, and context and localisation data. 
- **Synthetic Data:** For demonstration and privacy, synthetic datasets mimicking real-world alternative data are provided in the `tests/data/` directory.

> **Note:** Ensure compliance with all data privacy and regulatory requirements when working with real customer data.

## Project Structure

```
Credit-Risk-Probability-Model-for-Alternative-Data/
│
├── .dvc/                              # DVC configuration files
├── .github/                           # GitHub workflows and CI/CD
├── data/                              # Data directory
|   ├── raw/                           # Raw data files
|   └── processed/                     # Processed data files
├── models/                            # Directory for trained models
|   ├── best_model.pkl                 # Example trained model file
|   ...                                
├── notebooks/                         # Jupyter notebooks
|   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
|   ├── 02_Feature_Engineering.ipynb   # Feature engineering and WoE
|   ...                                
├── plots/                             # Plots and visualisations
├── scripts/                           # Data processing and modelling scripts
│   ├── api/                           # API scripts for deployment
│   │   ├── app.py                     # API application
│   │   └── python_api.py              # API script
│   ├── __init__.py                    # Package initialisation
│   ├── _01_EDA.py                     # EDA script
│   ├── _02_Feature_Engineering.py     # Feature engineering script
   ...                                 
├── test notebooks/                    # Test notebooks
│   ├── 01_Test_EDA.ipynb              # Test for EDA notebook
├── tests/                             # Unit and integration tests
│   ├── data/                          # Test data files
|   |   └──temp.csv
│   ├── test_02_feature_engineering.py 
│   ├── test_03_model_training.py      
│   ...                                
├── .dockerignore                      # Docker ignore file
├── .gitignore                         # Git ignore file
├── Dockerfile                         # Docker build file
├── README.md                          # Project documentation
├── docker-compose.yml                 # Docker Compose configuration
├── pytest.ini                         # pytest configuration
├── requirements-docker.txt            # Docker-specific dependencies
└── requirements.txt                   # Python dependencies

```

## Installation

### Prerequisites

- Python 3.8 or newer (recommended)
- `pip` (Python package manager)
- [DVC](https://dvc.org/) (for data version control)
- [Git](https://git-scm.com/)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nuhaminae/Credit-Risk-Probability-Model-for-Alternative-Data.git
   cd Credit-Risk-Probability-Model-for-Alternative-Data
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .credvenv
   # On Windows:
   .credvenv\Scripts\activate
   # On Unix/macOS:
   source .credvenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up DVC:**
   ```bash
   dvc pull
   ```

## Usage

### Data Preparation

- Place your data files (CSV or similar) in the `data/` directory.
- Review and update file paths in the notebooks or scripts as required.

### Running Notebooks

- Launch Jupyter Notebook:

  ```bash
  jupyter notebook
  ```

- Open and execute the notebooks in the `notebooks/` directory in sequence.

### Model Training via Scripts

- Run the main training and evaluate the trained model:
  ```bash
  python scripts/_04_Modelling_Tracking.py --data test/data/train.csv
  ```

### Deployment
Deployment scripts are provided for exposing the model as a RESTful API. The API accepts borrower data and returns a probability of default, supporting integration with external systems.

#### Run Locally (Without Docker)
- To launch the FastAPI server directly—ideal for development or quick testing:
  ```bash
   python scripts/app.py
  ```
- Once running, the API will be available at:
  - Swagger UI: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health

#### Run with Docker 
- To containerise and deploy the Credit Risk Scoring API:
```bash
# Step 1: Build the Docker image
docker build -t credit-risk-api .

# Step 2: Run the container, exposing port 8000
docker run -p 8000:8000 credit-risk-api
```

## Model Training and Evaluation
The project supports experimentation with multiple algorithms and hyperparameter tuning. Results are logged and visualised for comparison. Key evaluation metrics include:

- Area Under the ROC Curve (ROC-AUC)
- Precision, Recall, F1-score
- Confusion Matrix
- Feature Importance Charts

## Results and Interpretation

Results and analyses are documented in the `notebooks/` directory. Example outputs include:

- Comparative performance of models using feature engineering techniques.
- Insights into which alternative features are most predictive of credit default.
- Visualisations of model performance metrics.

## Testing

Continuous integration is set up using GitHub Actions (`.github/workflows/ci.yml`). This runs basic tests such as checking Python version, installing dependencies, and running pytests.

To run tests locally, ensure your environment is activated and use:

```bash
python --version
pytest tests/
```

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or suggestions.

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.


## Project Status
The project is completed, please check [commit history](https://github.com/nuhaminae/Credit-Risk-Probability-Model-for-Alternative-Data/commits?author=nuhaminae).

## Acknowledgements

Special thanks to the open-source community and data science practitioners whose tools and insights made this project possible.

---

For any questions or support, please open an issue or contact the maintainer at [Nuhamin](https://github.com/nuhaminae).
