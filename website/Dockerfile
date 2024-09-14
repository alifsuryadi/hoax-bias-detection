# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Show the contents of requirements.txt for debugging
RUN echo "Contents of requirements.txt:" && cat requirements.txt

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]
