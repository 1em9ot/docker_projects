ARG VERSION=latest
FROM implem/pleasanter:${VERSION}
# プロジェクトルートにある app_data_parameters を、
# pleasanter/ から見た相対パスでコピー
COPY ../app_data_parameters/ App_Data/Parameters/
# entrypoint.sh を、同一ディレクトリ内の pleasanter/entrypoint.sh からルートにコピー
COPY pleasanter/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
