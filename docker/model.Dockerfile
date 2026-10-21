FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt && useradd --no-create-home --uid 1000 byosm
COPY . /model
USER byosm
EXPOSE 8800
CMD ["python", "/model/byosm_rt.py", "serve", "/model", "--host", "0.0.0.0", "--port", "8800"]
