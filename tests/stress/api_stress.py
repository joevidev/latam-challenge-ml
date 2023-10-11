from locust import HttpUser, task
from datetime import datetime, timedelta

class StressUser(HttpUser):
    
    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas", 
                        "TIPOVUELO": "N", 
                        "MES": 3,
                        "Fecha_I": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        "Fecha_O": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ]
            }
        )


    @task
    def predict_latam(self):
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Grupo LATAM", 
                        "TIPOVUELO": "N", 
                        "MES": 3,
                        "Fecha_I": (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                        "Fecha_O": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ]
            }
        )