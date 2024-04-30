FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . /app

ENV FLASK_APP=app_iris.py
ENV FLASK_ENV=development
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=4000

CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]
