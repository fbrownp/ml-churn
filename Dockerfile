FROM python:3.11


WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# CMD ["python3" , "app.py"]
EXPOSE 8080

CMD ["uvicorn", "app:app", "--reload","--port", "8080", "--host", "0.0.0.0"]
