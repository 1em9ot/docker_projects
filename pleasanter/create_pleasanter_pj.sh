#!/bin/bash
# setup_pleasanter.sh
# PleasanterDocker プロジェクトの作成から初期セットアップ＆起動までを全自動で行うスクリプト
# ※公式マニュアル「Getting Started: Pleasanter Docker」に沿った内容です。

set -e

# -------------------------------
# 変数設定
# -------------------------------
PROJECT_ROOT="PleasanterDocker"
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
      # 接続文字列は Rds.json の内容を利用するため、環境変数による上書きは行いません

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
volumes:
  pg_data:
EOF

# -------------------------------
# サブフォルダーの作成（pleasanter, codedefiner, app_data_parameters）
# -------------------------------
echo "サブフォルダーを作成します…"
mkdir -p pleasanter codedefiner app_data_parameters

# -------------------------------
# pleasanter/entrypoint.sh の作成
# -------------------------------
echo "pleasanter/entrypoint.sh を作成します…"
cat > pleasanter/entrypoint.sh << 'EOF'
#!/bin/sh
# Pleasanter コンテナ起動前に Rds.json を再生成して正しい接続文字列を設定する

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
# コピー元：プロジェクトルートにある app_data_parameters を、Dockerfile のある pleasanter/ から見た相対パスでコピー
COPY ../app_data_parameters/ App_Data/Parameters/
# entrypoint.sh は pleasanter/ フォルダー内にあるので、ここでは "pleasanter/entrypoint.sh" を指定
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
# Dockerfile のある codedefiner/ から見た相対パスで、app_data_parameters をコピー
COPY ../app_data_parameters/ /app/Implem.Pleasanter/App_Data/Parameters/
ENTRYPOINT [ "dotnet", "Implem.CodeDefiner.dll" ]
EOF

# -------------------------------
# app_data_parameters/Rds.json の作成（接続文字列を固定値で記載）
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
# Pleasanter コンテナの起動
# -------------------------------
echo "Pleasanter コンテナをバックグラウンド起動します…"
docker compose up -d pleasanter

echo "全自動セットアップが完了しました。"
echo "ブラウザで http://localhost:8881 にアクセスして、Pleasanter のログイン画面を確認してください。"
