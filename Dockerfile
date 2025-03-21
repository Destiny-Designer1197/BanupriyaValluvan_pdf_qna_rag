# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all src files
COPY src/ /app/src
# Copy the .env file
COPY .env/ /app/.env

# copy the PDF files
COPY pdfs /app/pdfs

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Set the default command to run the main script
CMD ["python", "src/main.py", "--pdf_dir", "pdfs"]
