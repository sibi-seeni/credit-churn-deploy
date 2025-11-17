# Dockerfile

# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary scripts and saved model files
COPY predict.py .
COPY model.pkl .
COPY encoders.pkl .
COPY columns.json .

# Expose the port the Flask app runs on
EXPOSE 5001

# The command to run when the container starts
CMD ["python", "predict.py"]