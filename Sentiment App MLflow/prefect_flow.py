from prefect import flow, task
import subprocess

@task
def train_sentiment_model():
    subprocess.run(["python", "sentiment_mlflow.ipynb"], check=True)

@flow(name="Sentiment-MLflow-Pipeline")
def sentiment_pipeline():
    train_sentiment_model()

if __name__ == "__main__":
    sentiment_pipeline()
