import pandas as pd
import xgboost as xgb
from typing import Tuple, Union, List
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_FILENAME = "trained_model.xgb"

class DelayModel:
    def __init__(self):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=5.407)

        if os.path.exists(MODEL_FILENAME):
            try:
                self._model.load_model(MODEL_FILENAME)
                logging.info("Loaded the trained model.")
            except Exception as e:
                logging.error(f"Error loading the model: {e}")
                self._model = None
        else:
            logging.info("DelayModel initialized with a new model.")

    @staticmethod
    def get_period_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 19:
            return 'afternoon'
        else:
            return 'night'

    @staticmethod
    def is_high_season(date):
        if (date.month == 12 and date.day >= 15) or (date.month == 1) or (date.month == 2 and date.day <= 3) or \
           (date.month == 7 and (date.day >= 15 and date.day <= 31)) or (date.month == 9 and (date.day >= 11 and date.day <= 30)):
            return 1
        return 0

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        logging.info("Starting preprocessing.")
        # Convert Fecha-I to datetime
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])

        # Create period_day column
        data['period_day'] = data['Fecha-I'].dt.hour.apply(self.get_period_day)

        # Create high_season column
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)

        # Create min_diff and delay columns
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        data['delay'] = data['min_diff'].apply(lambda x: 1 if x > 15 else 0)

        expected_columns = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        tipo_vuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
        
        # Ensure all expected columns are present, if not, add them with zeros
        for col in expected_columns:
            if col not in opera_dummies.columns:
                opera_dummies[col] = 0
            if col not in tipo_vuelo_dummies.columns:
                tipo_vuelo_dummies[col] = 0
            if col not in mes_dummies.columns:
                mes_dummies[col] = 0

        features = pd.concat([
            opera_dummies[[col for col in expected_columns if 'OPERA_' in col]],
            tipo_vuelo_dummies[[col for col in expected_columns if 'TIPOVUELO_' in col]],
            mes_dummies[[col for col in expected_columns if 'MES_' in col]]
        ], axis=1)
        
        logging.info(f"Preprocessing completed. Features shape: {features.shape}")

        if target_column:
            target = data[[target_column]]  # Return target as DataFrame
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        logging.info("Starting training.")

        # Handle class imbalance by assigning weights
        class_weights = target['delay'].apply(lambda x: 3 if x == 1 else 1).values

        # Train the XGBClassifier_2 model
        self._model.fit(features, target.values.ravel(), sample_weight=class_weights)
        try:
            self._model.save_model(MODEL_FILENAME)
            logging.info("Model saved successfully.")
        except Exception as e:
            logging.error(f"Error saving the model: {e}")

    def predict(self, features: pd.DataFrame, threshold: float = 0.718) -> List[int]:
        logging.info("Starting prediction.")

        # Get probability predictions using the trained model
        prob_predictions = self._model.predict_proba(features)[:, 1]

        # Convert probability predictions to binary labels based on the threshold
        predictions = [1 if prob > threshold else 0 for prob in prob_predictions]

        logging.info(f"Prediction completed. Predicted {len(predictions)} values.")
        return predictions
