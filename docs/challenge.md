## Context

We need to operationalize the data science and machine learning work for the airport team. For this, we have decided to enable an API where they can query the delay prediction of a flight.

## Part I: Model Transcription and Enhancement (model.py)

The model.py file implements a `DelayModel` class that encapsulates the logic of the prediction model. This class is responsible for:

- Preprocessing the data, including feature engineering.
- Training the model using an XGBoost classifier.
- Predicting the delay probability of a flight.

### Model Selection

Through an exploratory analysis on the `exploration.ipynb` notebook, several models were trained, including different configurations of XGBoost and logistic regression. The model chosen for `model.py` was `xgb_model_2`, which uses the XGBoost classifier with an adjustment to the `scale_pos_weight` parameter to address class imbalance.

The choice of `xgb_model_2` was based on its ability to detect delays with superior recall compared to other models. Although it has a lower overall accuracy than some of the other models, it has a significantly better recall for the delay class. In an airport context, proactively detecting delays is essential. Early detection of delays allows the airport to take preventative action and improve the customer experience.

## Part II: Deploying the Model in an API (api.py)

The api.py file uses FastAPI to deploy the model as a service. The API has two main endpoints:

- `/health`: Verifies that the API is working correctly.
- `/predict`: Accepts flight data and returns the delay prediction.

The API includes validations to ensure that the input data is correct. For example, it validates that `FLIGHTTYPE` is `N` or `I` and that `MONTH` is in the range 1 to 12.

## Part III: Deployment on Google Cloud Platform

The app.yaml file provides the configuration required to deploy the application to App Engine on Google Cloud Platform (GCP). The configuration specifies the Python version, the entry point for the application, the instance class, and the auto-scaling configuration.

## Part IV: CI/CD Deployment

Two workflow files, ci.yml and cd.yml, are provided for continuous integration and continuous deployment respectively. These files define how the project is built, tested and deployed in GitHub Actions.

## Configuration and Deployment with Docker and Makefile.

The Dockerfile defines how the Docker container is built for the application. It ensures that the proper dependencies are installed and that the application starts correctly.

The Makefile provides commands to facilitate various tasks related to the development and testing of the project, including installing dependencies, running tests, and building the project.
