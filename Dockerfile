#Run this in the integrated terminal: docker build --progress=plain -t symtrain-assistant . 
# syntax=docker/dockerfile:1

#Creating the base image of lightweight Python.
FROM python:3.11

#Creating the environment for Python in Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

#Setting the working directory inside the container.
WORKDIR /app

#Installing the Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt

#Copying the rest of the repo.
COPY . .

#Exposing Streamlit’s default port.
EXPOSE 8501

#Running the Streamlit app.
CMD ["streamlit", "run", "symtrain_assistant/app.py", "--server.port=8501", "--server.address=0.0.0.0"]