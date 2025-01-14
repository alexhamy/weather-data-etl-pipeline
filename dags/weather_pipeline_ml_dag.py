import requests
import pandas as pd
import psycopg2
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Constants for Weather API and PostgreSQL
API_KEY = os.getenv("API_KEY")
CITY = "London"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
POSTGRES_CONN_ID = 'my_postgres_conn'

# Define file paths within the shared volume
RAW_DATA_PATH = "/opt/airflow/shared/weather_data_raw.csv"
PROCESSED_DATA_PATH = "/opt/airflow/shared/weather_data_processed.csv"
MODEL_PATH = "/opt/airflow/shared/weather_model.pkl"

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

dag = DAG(
    'weather_pipeline_ml',
    default_args=default_args,
    description='A pipeline to collect, store, process, and model weather data.',
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

# Step 3: Process Data
def process_weather_data():
    print("Processing weather data...")
    conn = BaseHook.get_connection(POSTGRES_CONN_ID)
    conn = psycopg2.connect(
        host=conn.host,
        dbname=conn.schema,
        user=conn.login,
        password=conn.password,
        port=conn.port
    )

    query = "SELECT * FROM weather;"
    weather_df = pd.read_sql_query(query, conn)

    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df['month'] = weather_df['date'].dt.month
    weather_df['day'] = weather_df['date'].dt.day
    weather_df['hour'] = weather_df['date'].dt.hour

    scaler = MinMaxScaler()
    numerical_columns = ['temperature', 'humidity', 'wind_speed']
    weather_df[numerical_columns] = scaler.fit_transform(weather_df[numerical_columns])

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    weather_df.to_csv(PROCESSED_DATA_PATH, index=False)
    conn.close()
    print("Weather data processed and saved.")

# Step 4: Train Machine Learning Model
def train_machine_learning_model():
    print("Training machine learning model...")
    weather_df = pd.read_csv(PROCESSED_DATA_PATH)

    X = weather_df[['temperature', 'humidity', 'wind_speed', 'month', 'day', 'hour']]
    y = weather_df['temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model Evaluation: R^2 Score = {score}")

    joblib.dump(model, MODEL_PATH)
    print("Model saved.")

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

process_data_task = PythonOperator(
    task_id='process_weather_data',
    python_callable=process_weather_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_machine_learning_model',
    python_callable=train_machine_learning_model,
    dag=dag,
)

# Task dependencies
collect_data_task >> save_data_task >> process_data_task >> train_model_task
