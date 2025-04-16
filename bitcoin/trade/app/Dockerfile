FROM python:3.11-slim
WORKDIR /app
# タイムゾーンを JST (Asia/Tokyo) に設定
RUN apt-get update && apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo 'Asia/Tokyo' > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Tokyo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8050
CMD ["python", "app.py"]
