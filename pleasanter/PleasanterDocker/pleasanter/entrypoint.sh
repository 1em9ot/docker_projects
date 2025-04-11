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
