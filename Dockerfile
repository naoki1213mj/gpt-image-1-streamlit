# ---------- ビルドステージ ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# 依存ファイルだけ先にコピーしてキャッシュ
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- ランタイムステージ ----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Streamlit の標準ポート
    PORT=8501

WORKDIR /app

# wheel を展開して軽量インストール
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*

# アプリのコードを最後にコピー
COPY . .

# 本番用設定
ENV STREAMLIT_SERVER_HEADLESS=true
#   └ Streamlit 1.25+ は headless 自動判定しますが安全のため指定

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port","8501","--server.address","0.0.0.0"]
