FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY serve/app.py /app/app.py
# SMOKE ONLY: bake the trained pack so CI/local can boot without Hub.
# Prod-shaped serve: do NOT rely on this. Set MODEL_SOURCE=hub + HF_MODEL_ID + HF_REVISION.
COPY model/ /model/

ENV MODEL_DIR=/model
ENV MODEL_SOURCE=local
ENV REQUIRE_TOKEN=1

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
