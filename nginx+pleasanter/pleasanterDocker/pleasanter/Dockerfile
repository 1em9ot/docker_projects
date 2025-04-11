ARG VERSION=latest
FROM implem/pleasanter:${VERSION}
# プロジェクトルートにある app_data_parameters を、
# pleasanter/ から見た相対パスでコピー
COPY ../app_data_parameters/ App_Data/Parameters/
# entrypoint.sh は pleasanter/ 配下にあるので、相対パスでコピー
COPY pleasanter/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
