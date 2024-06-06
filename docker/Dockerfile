FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install tensorboard

COPY src/ src/

ENTRYPOINT ["python", "src/main.py"]
