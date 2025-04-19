FROM python:3.11-slim
WORKDIR /app
# 日本時間設定＋git
RUN apt-get update && apt-get install -y tzdata git && \
    ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo 'Asia/Tokyo' > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Tokyo

# 依存ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app.py .

EXPOSE 8050
CMD ["python", "app.py"]
