from google.cloud import aiplatform

def create_run_pipeline(project_id, bucket_name, image_uri):
    # Vertex AI SDK ko initialize karna
    aiplatform.init(
        project=project_id, 
        location='us-central1',
        staging_bucket=f"gs://{bucket_name}"
    )

    # Custom Job Define karna
    job = aiplatform.CustomContainerTrainingJob(
        display_name="loan-approval-training-pipeline",
        container_uri=image_uri,
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
    )

    # Job ko Run karna
    print(f"Starting Vertex AI Training Job for Project: {project_id}")
    model = job.run(
        model_display_name="loan-approval-prediction-model",
        args=[f"--bucket_name={bucket_name}"],
        replica_count=1,
        machine_type="n1-standard-4",
        sync=True
    )

    # Model ko Endpoint par Deploy karna
    print("Deploying model to endpoint...")
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1
    )
    
    print(f"Model successfully deployed!")
    print(f"Endpoint Resource Name: {endpoint.resource_name}")

if __name__ == "__main__":
    PROJECT_ID = "neural-quarter-490713-b0"
    BUCKET_NAME = "my-mlops-bucket-ritik"
    IMAGE_URI = f"us-central1-docker.pkg.dev/{PROJECT_ID}/ml-repo/loan-trainer:latest"
    
    create_run_pipeline(PROJECT_ID, BUCKET_NAME, IMAGE_URI)