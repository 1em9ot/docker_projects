FROM python:3.11-slim
WORKDIR /myhealth
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8060
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python","/myhealth/dashboard/app.py"]
