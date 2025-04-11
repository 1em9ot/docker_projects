FROM implem/pleasanter:codedefiner
# Dockerfile のある codedefiner/ から見た相対パスで、app_data_parameters をコピー
COPY ../app_data_parameters/ /app/Implem.Pleasanter/App_Data/Parameters/
ENTRYPOINT [ "dotnet", "Implem.CodeDefiner.dll" ]
