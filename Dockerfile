FROM python:3.9-alpine

# Install build dependencies
RUN apt-get update &&  \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libhdf5-dev python-opencv && \
    apt-get install -y libgomp1
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install paddlepaddle



COPY . /app
WORKDIR /app
EXPOSE 5000


CMD ["python", "app.py"]