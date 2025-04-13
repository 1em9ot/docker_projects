#!/bin/bash
# setup_pleasanter_direct.sh
# PleasanterDocker プロジェクトのセットアップを全自動で行い、
# Pleasanter コンテナを直接外部に公開（ホスト側の 8881 番ポートでアクセス）するスクリプト
#
# 【重要】
# ASP.NET Core アプリ側（Startup.cs または Program.cs）には、可能であれば
# 以下の設定を追加してください。（ソースコードからビルドする場合のみ自動パッチも試みます）
#
#   using Microsoft.AspNetCore.HttpOverrides;
#   // … 必要な using を追加
#   var forwardedOptions = new ForwardedHeadersOptions {
#       ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto
#   };
#   forwardedOptions.KnownNetworks.Clear();
#   forwardedOptions.KnownProxies.Clear();
#   app.UseForwardedHeaders(forwardedOptions);
#
#   // ※ サブディレクトリを利用しない場合は UsePathBase は不要です。
#
# 【ホスト側 /etc/hosts の設定（WSL の場合）】
# WSL 環境の場合、/etc/hosts に "ken2025" のエントリーがなければ自動で追加します。
# Windows 側の hosts (例: C:\Windows\System32\drivers\etc\hosts) は手動で設定してください。
#
# このスクリプトは、Nginx やサブディレクトリ（/myhomesite）を使わずに、
# Pleasanter コンテナを直接 8881 番ポート経由で公開し、http://ken2025:8881/ でアクセスできる環境を構築します。

set -e

# --- WSL 環境の場合の /etc/hosts 設定 ---
if grep -qi "microsoft" /proc/version; then
    echo "WSL 環境が検出されました。/etc/hosts を確認します..."
    if ! grep -qi "ken2025" /etc/hosts; then
        echo "127.0.0.1    ken2025" | sudo tee -a /etc/hosts
        echo "Entry added to /etc/hosts."
    else
        echo "/etc/hosts に既に 'ken2025' のエントりがあります。"
    fi
fi

# -------------------------------
# 基本ディレクトリ／変数設定
# -------------------------------
BASE_DIR=$(pwd)
PROJECT_ROOT="${BASE_DIR}/pleasanterDocker"

# -------------------------------
# 既存のプロジェクトフォルダが存在する場合は削除
# -------------------------------
if [ -d "${PROJECT_ROOT}" ]; then
    echo "既存の '${PROJECT_ROOT}' フォルダが見つかりました。全てを削除して作り直します..."
    cd "${PROJECT_ROOT}" || exit 1
    docker compose down # ボリュームまで完全削除なら、docker compose down --volumes

    cd "${BASE_DIR}" || exit 1
    rm -rf "${PROJECT_ROOT}"
fi

# -------------------------------
# プロジェクトルートフォルダの作成
# -------------------------------
echo "プロジェクトフォルダ '${PROJECT_ROOT}' を作成します..."
mkdir -p "${PROJECT_ROOT}" || { echo "フォルダ作成に失敗しました。"; exit 1; }
cd "${PROJECT_ROOT}" || exit 1

# -------------------------------
# docker-compose.yml の作成
# -------------------------------
# ここでは Pleasanter コンテナを直接外部に 8881 番で公開するため、
# nginx サービスは削除し、pleasanter サービスのポートマッピングを "8881:8080" に設定します。
echo "docker-compose.yml を作成します..."
cat > docker-compose.yml <<EOF
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
      ASPNETCORE_PATHBASE: ""
      # サブディレクトリ利用しないので UsePathBase は不要です。
      # ※ アプリ側で必要なら Forwarded Headers の設定は行ってください。

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
echo "サブフォルダーを作成します..."
mkdir -p pleasanter codedefiner app_data_parameters

# -------------------------------
# オプション: ASP.NET Core アプリソースパッチ (Startup.cs の自動修正例)
# -------------------------------
# ※ 以下はソースコードからビルドする場合の例です。該当ファイル (例: src/Startup.cs) が存在する場合のみ動作します。
if [ -f "src/Startup.cs" ]; then
    echo "Startup.cs が見つかりました。必要な設定を自動追加します..."
    if ! grep -q "app.UsePathBase(\"/\");" src/Startup.cs; then
        # サブディレクトリを使用しない直接公開の場合、UsePathBase は "" でよいので、ForwardedHeaders の設定だけ挿入する例
        sed -i '/app.UseRouting()/i\
    var forwardedOptions = new ForwardedHeadersOptions {\
        ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto\
    };\
    forwardedOptions.KnownNetworks.Clear();\
    forwardedOptions.KnownProxies.Clear();\
    app.UseForwardedHeaders(forwardedOptions);' src/Startup.cs
        echo "ForwardedHeaders の設定を追加しました。"
    else
        echo "Startup.cs に必要な設定は既に追加されています。"
    fi
fi

# -------------------------------
# pleasanter/entrypoint.sh の作成
# -------------------------------
echo "pleasanter/entrypoint.sh を作成します..."
cat > pleasanter/entrypoint.sh <<EOF
#!/bin/sh
# Pleasanter コンテナ起動前に、Rds.json を再生成して正しい接続文字列を設定する

RDS_FILE="App_Data/Parameters/Rds.json"

echo "Rds.json を再生成します..."
cat > "\$RDS_FILE" <<EOL
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
echo "pleasanter/Dockerfile を作成します..."
cat > pleasanter/Dockerfile <<EOF
ARG VERSION=latest
FROM implem/pleasanter:\${VERSION}
# プロジェクトルートにある app_data_parameters を、pleasanter/ から見た相対パスでコピー
COPY ../app_data_parameters/ App_Data/Parameters/
# entrypoint.sh は pleasanter/ 配下にあるので、相対パスでコピー
COPY pleasanter/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
EOF

# -------------------------------
# codedefiner/Dockerfile の作成
# -------------------------------
echo "codedefiner/Dockerfile を作成します..."
cat > codedefiner/Dockerfile <<EOF
FROM implem/pleasanter:codedefiner
# プロジェクトルートにある app_data_parameters を、codedefiner/ から見た相対パスでコピー
COPY ../app_data_parameters/ /app/Implem.Pleasanter/App_Data/Parameters/
ENTRYPOINT [ "dotnet", "Implem.CodeDefiner.dll" ]
EOF

# -------------------------------
# app_data_parameters/Rds.json の作成
# -------------------------------
echo "app_data_parameters/Rds.json を作成します..."
cat > app_data_parameters/Rds.json <<EOF
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
find . -type d

echo "Docker イメージをビルドします…"
docker compose build --no-cache --pull

echo "CodeDefiner を実行してデータベース初期化を行います…"
docker compose run --rm codedefiner _rds /l "ja" /z "Asia/Tokyo"

echo "全サービスをバックグラウンド起動します…"
docker compose up -d

echo "全自動セットアップが完了しました。"
echo "ホスト側では直接 Pleasanter コンテナが公開され、"
echo "ブラウザで http://ken2025:8881/ にアクセスして、Pleasanter のログイン画面を確認してください。"
