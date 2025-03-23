# $DEL_BEGIN
# syntax=docker/dockerfile:1

FROM python:3.10.6-slim

# # Set the working directory in the container
# WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY vinylitics vinylitics
COPY setup.py setup.py
COPY neighbors.dill neighbors.dill
COPY pca.dill pca.dill
COPY preproc_pipe.dill preproc_pipe.dill
COPY raw_data/dataframe_2.csv raw_data/dataframe_2.csv
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# # Create a non-root user and switch to it for security
# RUN useradd -m appuser
# USER appuser

# # Make port 8000 available to the world outside this container
# EXPOSE 8000

# Run the application when the container launches
CMD uvicorn vinylitics.api.fast:app --host 0.0.0.0 --port $PORT

# $DEL_END
