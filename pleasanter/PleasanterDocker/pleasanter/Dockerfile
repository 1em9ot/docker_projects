ARG VERSION=latest
FROM implem/pleasanter:${VERSION}
# コピー元：プロジェクトルートにある app_data_parameters を、Dockerfile のある pleasanter/ から見た相対パスでコピー
COPY ../app_data_parameters/ App_Data/Parameters/
# entrypoint.sh は pleasanter/ フォルダー内にあるので、ここでは "pleasanter/entrypoint.sh" を指定
COPY pleasanter/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
