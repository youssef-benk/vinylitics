
#################### PACKAGE ACTIONS ###################
run_api:
	uvicorn vinylitics.api.fast:app --reload
reinstall_package:
	@pip uninstall -y vinylitics || :
	@pip install -e .

run_main:
	python vinylitics/interface/main.py

run_recommender:
	python vinylitics/preproc/recommender.py


##################### TESTS #####################

test_api_root:
	pytest \
	tests/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api/test_endpoints.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \


# In case you are using an Apple Silicon, before pushing on the cloud:

docker_build_local:
	docker build --tag=$(GAR_IMAGE):dev .

docker_run_local:
	docker run \
		-e PORT=8000 -p 8000:8000 \
		--env-file .env \
		$(GAR_IMAGE):dev

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p 8000:8000 \
		--env-file .env \
		$(GAR_IMAGE):dev \
		bash

# Cloud images - linux/amd64 architecture is needed

gar_creation:
	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
	gcloud artifacts repositories create ${GAR_REPO} --repository-format=docker \
		--location=${GCP_REGION} --description="Repository for storing ${GAR_REPO} images"

docker_build:
	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod .

docker_push:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod

docker_run:
	docker run -e PORT=8000 -p 8000:8000 --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod

docker_interactive:
	docker run -it --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod /bin/bash

docker_deploy:
	gcloud run deploy --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod --memory ${GAR_MEMORY} --region ${GCP_REGION}

# Buckets and bigquery
reset_bq_files:
	-bq rm -f --project_id ${GCP_PROJECT} ${BQ_DATASET}.dataframe_2
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.dataframe_2

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}
