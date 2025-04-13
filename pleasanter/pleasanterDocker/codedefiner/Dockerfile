FROM implem/pleasanter:codedefiner
# プロジェクトルートにある app_data_parameters を、codedefiner/ から見た相対パスでコピー
COPY ../app_data_parameters/ /app/Implem.Pleasanter/App_Data/Parameters/
ENTRYPOINT [ "dotnet", "Implem.CodeDefiner.dll" ]
