FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /ood_slam

# Copy the training script, configuration file, and requirements into the container
COPY models/ ./models/
COPY data/ ./data/
COPY utils/ ./utils/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY requirements.txt ./
COPY main.py ./
# Install additional Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to run the training script with the configuration file
# ENTRYPOINT ["python3", "scripts/train.py", "--config", "config/default.json"]
ENTRYPOINT ["bash"]