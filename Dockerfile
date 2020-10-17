FROM python:3.8

MAINTAINER "akbir.94@gmail.com"
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]