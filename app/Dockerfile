FROM python:3.11-slim-buster
# FROM python:3.11-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
#["flask", "run", "--host=0.0.0.0", "--port=5000"]
CMD [ "python", "app.py" ] 