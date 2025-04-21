FROM python:3.11-slim
WORKDIR /myhealth
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*
COPY . .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8060
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python","/myhealth/dashboard/app.py"]
