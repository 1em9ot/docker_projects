#!/bin/bash
set -e

#-------------------------------------------------------
# このスクリプトは /home/user/docker_projects 直下で実行すること
#-------------------------------------------------------
if [ "$(basename "$PWD")" != "docker_projects" ]; then
    echo "Error: このスクリプトは /home/user/docker_projects 直下で実行してください。"
    exit 1
fi

#-------------------------------------------------------
# .git の存在確認と、存在しなければ /home/user からコピーする
#-------------------------------------------------------
if [ ! -d ".git" ]; then
    if [ -d "../.git" ]; then
        echo ".git が見つかりません。/home/user から .git をコピーします..."
        cp -r ../.git .
    else
        echo "Error: docker_projects と /home/user の両方に .git が見つかりません。"
        exit 1
    fi
fi

#-------------------------------------------------------
# バックアップ: 念のため .git のバックアップを作成
#-------------------------------------------------------
echo "バックアップ: 現行の .git フォルダを .git_backup にコピーします..."
cp -r .git .git_backup

#-------------------------------------------------------
# 不要なサブモジュール情報の削除 (.gitmodules と .git/config)
#-------------------------------------------------------
if [ -f ".gitmodules" ]; then
    echo ".gitmodules から mystic, mystics 関連の設定を削除します..."
    sed -i '/\[submodule "mystic"\]/,/^\[/d' .gitmodules
    sed -i '/\[submodule "mystics"\]/,/^\[/d' .gitmodules
fi

echo ".git/config から mystic, mystics 関連のサブモジュール設定を削除します..."
sed -i '/\[submodule "mystic"\]/,/^\[/d' .git/config || true
sed -i '/\[submodule "mystics"\]/,/^\[/d' .git/config || true

echo ".git/modules 以下にある mystic, mystics ディレクトリを削除します..."
rm -rf .git/modules/mystic .git/modules/mystics

#-------------------------------------------------------
# git-filter-repo の存在確認
#-------------------------------------------------------
if ! command -v git-filter-repo &> /dev/null; then
    echo "Error: git-filter-repo がインストールされていません。"
    echo "Conda 環境の場合は、次のコマンドでインストールできます:"
    echo "  conda install -c conda-forge git-filter-repo"
    exit 1
fi

#-------------------------------------------------------
# リポジトリの履歴書き換え: もともとモノレポ時の docker_projects/ プレフィックスを除去
#-------------------------------------------------------
echo "git-filter-repo により、コミット履歴から docker_projects/ のプレフィックスを削除します..."
git filter-repo --path-rename "docker_projects/":"" --force

echo "完了：/home/user/docker_projects 内のリポジトリが、不要なサブモジュール設定を削除し、"
echo "       docker_projects/ のプレフィックスが除去された状態に更新されました。"
