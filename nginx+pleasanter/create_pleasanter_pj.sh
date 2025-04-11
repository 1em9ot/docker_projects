#!/bin/bash
# setup_pleasanter_with_nginx.sh
# PleasanterDocker プロジェクトの作成から初期セットアップ＆起動までを全自動で行うスクリプト
# 外部から http://ken2025/myhomesite でアクセスできるように、Nginx をリバースプロキシとして利用する
# ※公式マニュアル「Getting Started: Pleasanter Docker」に沿った内容です

set -e

# -------------------------------
# 変数設定
# -------------------------------
PROJECT_ROOT="pleasanterDocker"
SERVICE_NAME="Implem.Pleasanter"  # ※ 必要に応じて変更してください
POSTGRES_PASSWORD="MyStrongPostgresPass!"
OWNER_PASSWORD="OwnerPass123!"
USER_PASSWORD="UserPass123!"

# -------------------------------
# 既存のプロジェクトフォルダが存在する場合削除
# -------------------------------
if [ -d "${PROJECT_ROOT}" ]; then
    echo "既存の '${PROJECT_ROOT}' フォルダが見つかりました。削除します…"
    rm -rf "${PROJECT_ROOT}"
fi

# -------------------------------
# プロジェクトルートフォルダの作成
# -------------------------------
echo "プロジェクトフォルダ '${PROJECT_ROOT}' を作成します…"
mkdir -p "${PROJECT_ROOT}" || { echo "フォルダ作成に失敗しました。"; exit 1; }
cd "${PROJECT_ROOT}" || exit 1

# -------------------------------
# docker-compose.yml の作成
# -------------------------------
echo "docker-compose.yml を作成します…"
cat > docker-compose.yml << 'EOF'
services:
  db:
    image: postgres:16
    container_name: pleasanter_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: "postgres"
      POSTGRES_PASSWORD: "MyStrongPostgresPass!"
      POSTGRES_DB: "postgres"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  pleasanter:
    build:
      context: .
      dockerfile: pleasanter/Dockerfile
    container_name: pleasanter_app
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
    ports:
      - "8881:8080"
    environment:
      ASPNETCORE_ENVIRONMENT: "Development"
      ASPNETCORE_PATHBASE: "/myhomesite"

  codedefiner:
    build:
      context: .
      dockerfile: codedefiner/Dockerfile
    container_name: pleasanter_codedefiner
    restart: "no"
    depends_on:
      db:
        condition: service_healthy
    environment:
      ASPNETCORE_ENVIRONMENT: "Development"

  nginx:
    image: nginx:latest
    container_name: nginx_proxy
    restart: unless-stopped
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - pleasanter

networks:
  default:
    name: pleasanterDocker_default

volumes:
  pg_data:
    name: pleasanterDocker_pg_data
EOF

# -------------------------------
# サブフォルダーの作成（pleasanter, codedefiner, app_data_parameters）
# -------------------------------
echo "サブフォルダーを作成します…"
mkdir -p pleasanter codedefiner app_data_parameters

# -------------------------------
# nginx.conf の作成
# -------------------------------
echo "nginx.conf を作成します…"
cat > nginx.conf << 'EOF'
worker_processes 1;
events { worker_connections 1024; }
http {
  server {
    listen 80;
    server_name ken2025;
    
    location /myhomesite/ {
      proxy_pass http://pleasanter_app:8080/myhomesite/;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
    }
  }
}
EOF

# -------------------------------
# pleasanter/entrypoint.sh の作成（Rds.json 再生成用）
# -------------------------------
echo "pleasanter/entrypoint.sh を作成します…"
cat > pleasanter/entrypoint.sh << 'EOF'
#!/bin/sh
# Pleasanter コンテナ起動前に、Rds.json を再生成して正しい接続文字列を設定する

RDS_FILE="App_Data/Parameters/Rds.json"

echo "Rds.json を再生成します…"
cat > "$RDS_FILE" <<EOL
{
  "Dbms": "PostgreSQL",
  "Provider": "Local",
  "SaConnectionString": "Server=db;Port=5432;Database=postgres;UID=postgres;PWD=MyStrongPostgresPass!",
  "OwnerConnectionString": "Server=db;Port=5432;Database=Implem.Pleasanter;UID=Implem.Pleasanter_Owner;PWD=OwnerPass123!",
  "UserConnectionString": "Server=db;Port=5432;Database=Implem.Pleasanter;UID=Implem.Pleasanter_User;PWD=UserPass123!",
  "SqlCommandTimeOut": 600,
  "MinimumTime": 3,
  "DeadlockRetryCount": 4,
  "DeadlockRetryInterval": 1000,
  "DisableIndexChangeDetection": true,
  "SysLogsSchemaVersion": 1
}
EOL

echo "Rds.json の再生成が完了しました。"
exec dotnet Implem.Pleasanter.dll
EOF
chmod +x pleasanter/entrypoint.sh

# -------------------------------
# pleasanter/Dockerfile の作成
# -------------------------------
echo "pleasanter/Dockerfile を作成します…"
cat > pleasanter/Dockerfile << 'EOF'
ARG VERSION=latest
FROM implem/pleasanter:${VERSION}
# プロジェクトルートにある app_data_parameters を、
# pleasanter/ から見た相対パスでコピー
COPY ../app_data_parameters/ App_Data/Parameters/
# entrypoint.sh は pleasanter/ 配下にあるので、相対パスでコピー
COPY pleasanter/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
EOF

# -------------------------------
# codedefiner/Dockerfile の作成
# -------------------------------
echo "codedefiner/Dockerfile を作成します…"
cat > codedefiner/Dockerfile << 'EOF'
FROM implem/pleasanter:codedefiner
# プロジェクトルートにある app_data_parameters を、
# codedefiner/ から見た相対パスでコピー
COPY ../app_data_parameters/ /app/Implem.Pleasanter/App_Data/Parameters/
ENTRYPOINT [ "dotnet", "Implem.CodeDefiner.dll" ]
EOF

# -------------------------------
# app_data_parameters/Rds.json の作成（固定値で記載）
# -------------------------------
echo "app_data_parameters/Rds.json を作成します…"
cat > app_data_parameters/Rds.json << 'EOF'
{
  "Dbms": "PostgreSQL",
  "Provider": "Local",
  "SaConnectionString": "Server=db;Port=5432;Database=postgres;UID=postgres;PWD=MyStrongPostgresPass!",
  "OwnerConnectionString": "Server=db;Port=5432;Database=Implem.Pleasanter;UID=Implem.Pleasanter_Owner;PWD=OwnerPass123!",
  "UserConnectionString": "Server=db;Port=5432;Database=Implem.Pleasanter;UID=Implem.Pleasanter_User;PWD=UserPass123!",
  "SqlCommandTimeOut": 600,
  "MinimumTime": 3,
  "DeadlockRetryCount": 4,
  "DeadlockRetryInterval": 1000,
  "DisableIndexChangeDetection": true,
  "SysLogsSchemaVersion": 1
}
EOF

echo "プロジェクトディレクトリの作成が完了しました。"
echo "作成したディレクトリ構成:"
find . -type d

# -------------------------------
# Docker イメージのビルド（キャッシュクリア）
# -------------------------------
echo "Docker イメージをビルドします…"
docker compose build --no-cache --pull

# -------------------------------
# CodeDefiner による初期セットアップの実行
# -------------------------------
echo "CodeDefiner を実行してデータベース初期化を行います…"
docker compose run --rm codedefiner _rds /l "ja" /z "Asia/Tokyo"

# -------------------------------
# 全サービスの起動
# -------------------------------
echo "全サービスをバックグラウンド起動します…"
docker compose up -d

echo "全自動セットアップが完了しました。"
echo "ブラウザで http://ken2025/myhomesite にアクセスして、Pleasanter のログイン画面を確認してください。"
