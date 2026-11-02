# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install build tools needed to compile some python packages
RUN apt-get update && apt-get install -y gcc build-essential

# Copy the requirements file
COPY requirements.txt .

# Install packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy the application source code and data
COPY ./textwave /app/textwave
COPY ./storage /app/storage

# Make port 5000 available
EXPOSE 5000

# FIX: Use python's module (-m) flag to run the app. This is the most robust method.
CMD ["python", "-u", "-m", "textwave.app"]