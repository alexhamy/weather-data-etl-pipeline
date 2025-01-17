import requests
import pandas as pd
import psycopg2
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
import os

# Constants for Weather API and PostgreSQL
API_KEY = os.getenv("API_KEY")
CITY = "London"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
POSTGRES_CONN_ID = "my_postgres_conn"

# Define file paths within the shared volume
RAW_DATA_PATH = "/opt/airflow/shared/weather_data_raw.csv"

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

dag = DAG(
    'weather_pipeline',
    default_args=default_args,
    description='A pipeline to collect, store, and process weather data.',
    schedule_interval='@daily',
    catchup=False,
)

# Step 1: Data Collection Task
def collect_weather_data():
    print("Collecting weather data...")
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        weather_data = {
            "city": CITY,
            "date": datetime.now(),
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        weather_df = pd.DataFrame([weather_data])
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        weather_df.to_csv(RAW_DATA_PATH, index=False, mode='a', header=not os.path.exists(RAW_DATA_PATH))
        print("Weather data collected and saved to CSV.")
    else:
        print(f"Failed to fetch weather data. Status Code: {response.status_code}")

# Step 2: Store Data to PostgreSQL
def save_to_postgres():
    print("Saving data to PostgreSQL...")
    conn = BaseHook.get_connection(POSTGRES_CONN_ID)
    conn = psycopg2.connect(
        host=conn.host,
        dbname=conn.schema,
        user=conn.login,
        password=conn.password,
        port=conn.port
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            id SERIAL PRIMARY KEY,
            city VARCHAR(50),
            date TIMESTAMP,
            temperature FLOAT,
            humidity FLOAT,
            wind_speed FLOAT
        );
    """)
    conn.commit()

    weather_df = pd.read_csv(RAW_DATA_PATH)

    for _, row in weather_df.iterrows():
        cursor.execute("""
            INSERT INTO weather (city, date, temperature, humidity, wind_speed)
            VALUES (%s, %s, %s, %s, %s);
        """, (row["city"], row["date"], row["temperature"], row["humidity"], row["wind_speed"]))

    conn.commit()
    cursor.close()
    conn.close()
    print("Data saved to PostgreSQL.")

# Define tasks
collect_data_task = PythonOperator(
    task_id='collect_weather_data',
    python_callable=collect_weather_data,
    dag=dag,
)

save_data_task = PythonOperator(
    task_id='save_to_postgres',
    python_callable=save_to_postgres,
    dag=dag,
)

# Task dependencies
collect_data_task >> save_data_task
