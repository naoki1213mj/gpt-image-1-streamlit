# ---- ベースイメージ：Slim な公式 Python
FROM python:3.12-slim AS base

# ---- システム依存ライブラリを最小限インストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- 作業ディレクトリを作成
WORKDIR /app

# ---- Python 依存を先に入れてキャッシュ利用
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- アプリソースをコピー
COPY . .

# ---- Streamlit 実行ポート（App Service では WEBSITES_PORT で上書きしても OK）
ENV PORT=8501
EXPOSE 8501

# ---- エントリポイント
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
