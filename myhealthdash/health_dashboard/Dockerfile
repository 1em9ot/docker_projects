FROM python:3.11-slim

# ── 基本セットアップ ─────────────────────────
WORKDIR /myhealth
COPY requirements.txt .

# 依存ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 日本語フォント & LFS など追加パッケージ
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates \
      fonts-noto-cjk libgl1 libglib2.0-0 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# アプリ一式をコピー
COPY . .

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8060
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/myhealth/dashboard/app.py"]
