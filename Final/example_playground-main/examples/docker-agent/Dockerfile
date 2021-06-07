FROM python:3.6

ADD ./examples/docker-agent /agent

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
COPY ./pommerman/torch-1.8.1-cp36-cp36m-manylinux1_x86_64.whl /pommerman
RUN cd /pommerman && pip install torch-1.8.1-cp36-cp36m-manylinux1_x86_64.whl && pip install . -i https://mirrors.aliyun.com/pypi/simple --default-timeout=6000
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
ADD ./examples/docker-agent /agent
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
