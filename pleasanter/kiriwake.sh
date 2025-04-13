#!/bin/bash
# extended_troubleshoot.sh
# Nginx / Pleasanter 環境における詳細な情報収集を自動で行う統合スクリプト
# 以下の項目を順次実行して出力します:
#   1. コンテナ一覧の確認
#   2. Docker ネットワーク情報の確認
#   3. docker-compose.yml の内容確認
#   4. nginx.conf の内容確認
#   5. Nginx コンテナ内 /etc/nginx/html の内容確認
#   6. Nginx コンテナ内からの内部接続テスト
#   7. ホストからの curl アクセステスト
#   8. ホスト側の /etc/hosts 内容の確認（対象ホスト名が設定されているか）
#
# 使い方:
#   chmod +x extended_troubleshoot.sh
#   ./extended_troubleshoot.sh

set -e

# ここで NG_PORT 変数も設定（必要に応じて変更）
NG_PORT="8080"

echo "============================"
echo "1. コンテナ一覧の確認"
echo "----------------------------"
docker ps --format '{{.Names}}'
echo ""

echo "============================"
echo "2. Docker ネットワーク情報の確認"
echo "----------------------------"
docker network inspect pleasanterDocker_default || echo "ネットワーク情報取得に失敗"
echo ""

echo "============================"
echo "3. docker-compose.yml の内容確認"
echo "----------------------------"
if [ -f docker-compose.yml ]; then
    cat docker-compose.yml
else
    echo "docker-compose.yml が見つかりません。"
fi
echo ""

echo "============================"
echo "4. nginx.conf の内容確認"
echo "----------------------------"
if [ -f nginx.conf ]; then
    cat nginx.conf
else
    echo "nginx.conf が見つかりません。"
fi
echo ""

echo "============================"
echo "5. Nginx コンテナ内 /etc/nginx/html の内容確認"
echo "----------------------------"
# Nginx コンテナの名前は nginx_proxy または nginx としている可能性があるので両方チェック
if docker ps --format '{{.Names}}' | grep -q "nginx_proxy"; then
    docker exec -it nginx_proxy ls -l /etc/nginx/html || echo "/etc/nginx/html の確認に失敗"
elif docker ps --format '{{.Names}}' | grep -q "nginx"; then
    docker exec -it nginx ls -l /etc/nginx/html || echo "/etc/nginx/html の確認に失敗"
else
    echo "Nginx コンテナが見つかりません。"
fi
echo ""

echo "============================"
echo "6. Nginx コンテナ内からの内部接続テスト"
echo "----------------------------"
echo ">> curl -v http://pleasanter:8080/myhomesite/"
if docker ps --format '{{.Names}}' | grep -q "nginx_proxy"; then
    docker exec -it nginx_proxy curl -v http://pleasanter:8080/myhomesite/ || echo "内部 curl テスト /myhomesite/ でエラー"
else
    echo "Nginx コンテナ（nginx_proxy）が見つかりません。"
fi
echo ""

echo ">> curl -v http://pleasanter:8080/myhomesite/styles/plugins/normalize.css"
if docker ps --format '{{.Names}}' | grep -q "nginx_proxy"; then
    docker exec -it nginx_proxy curl -v http://pleasanter:8080/myhomesite/styles/plugins/normalize.css || echo "内部 curl テスト 静的ファイルでエラー"
else
    echo "Nginx コンテナ（nginx_proxy）が見つかりません。"
fi
echo ""

echo "============================"
echo "7. ホストからのアクセステスト（curl）"
echo "----------------------------"
echo ">> ホスト: curl -v http://127.0.0.1:${NG_PORT}/myhomesite/  (注意: URL中の 'myhomesite' を実際のパスに合わせて修正)"
# ※ URL中に誤字があれば、実際のパス (例: /myhomesite/ ではなく /myhomesite/ になっているか確認)
curl -v http://127.0.0.1:${NG_PORT}/myhomesite/ || echo "ホストからの curl /myhomesite/ でエラー"
echo ""

echo ">> ホスト: curl -v http://ken2025:${NG_PORT}/myhomesite/"
curl -v http://ken2025:${NG_PORT}/myhomesite/ || echo "ホストからの curl /myhomesite/ (ken2025) でエラー"
echo ""

echo ">> ホスト: curl -v http://ken2025:${NG_PORT}/styles/plugins/normalize.css"
curl -v http://ken2025:${NG_PORT}/styles/plugins/normalize.css || echo "ホストからの curl 静的ファイルでエラー"
echo ""

echo "============================"
echo "8. ホスト側の /etc/hosts の内容確認"
echo "----------------------------"
echo "以下に 'ken2025' のエントリーがあるか確認してください:"
grep -i "ken2025" /etc/hosts || echo "ホストの /etc/hosts に 'ken2025' のエントリーが見つかりません"
echo ""

echo "============================"
echo "情報収集完了。"
echo "上記出力結果のうち、どの段階で期待する動作と異なるか、またエラーメッセージがあるか教えてください。"
