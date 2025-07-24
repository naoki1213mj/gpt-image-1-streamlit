# GPT-image-1 × Streamlit 画像生成AIアプリ (Azureデプロイ対応)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要 📖

このリポジトリは、OpenAIの最新画像生成モデル **GPT-image-1** と、Python製のWeb UIフレームワーク **Streamlit** を用いて開発された画像生成アプリケーションのサンプル実装です。

ローカル環境での実行はもちろん、Dockerコンテナ化、そしてAzureのPaaS (**App Service for Containers** / **Azure Container Apps**) へのデプロイまでを網羅したフルスタックな構成例となっています。

特に、エンタープライズ利用を想定した **Microsoft Entra ID (旧Azure AD) による認証**や、**Azure Key Vault を用いた安全なシークレット管理** (Managed Identity連携) の実装方法を含んでいる点が特徴です。

**詳細な解説やAzureへのデプロイ手順、アプリの動作イメージ**については、以下のZenn記事をご参照ください。

**📄 Zenn記事:** [GPT-image-1 × Streamlit で作る画像生成AIアプリ - Entra ID認証連携からAzureデプロイまで完全解説](https://zenn.dev/chips0711/articles/28eee04b8f2cfd)

## 主な機能 ✨

* **Text-to-Image (txt2img):** テキストプロンプトから画像を生成します。
* **Image-to-Image (img2img):** アップロードした画像を基に、プロンプトに従って画像を編集・生成します。(※Azure OpenAI Serviceでの実装には注意点あり。詳細はZenn記事参照)
* **パラメータ調整UI:** 画像サイズ (`size`)、品質 (`quality`)、スタイル (`style`) などの生成オプションをStreamlitのUIから直感的に設定できます。
* **バックエンド切り替え:** 環境変数により、バックエンドを **OpenAI API** または **Azure OpenAI Service API** に切り替え可能です。
* **認証:**
  * Azureデプロイ時: Microsoft Entra ID 連携 (Easy Auth)
  * ローカル実行時: Streamlit標準のOAuth (`st.login`) などで代替可能 (要設定)
* **UI/UX:** プロンプト例ボタン、初回チュートリアル (`st.dialog`)、処理中スピナー (`st.spinner`) などを実装。

## 技術スタック 🛠️

* **言語:** Python 3.12
* **Webフレームワーク:** Streamlit
* **AIモデル:** GPT-image-1 (OpenAI / Azure OpenAI Service)
* **Pythonライブラリ:** openai, azure-identity, python-dotenv, Pillow, requests, streamlit
* **パッケージ管理:** uv (推奨)
* **コンテナ:** Docker
* **クラウド (Azure):**
  * Azure Container Registry (ACR)
  * Azure App Service for Containers
  * Azure Container Apps (ACA)
  * Azure Key Vault
  * Microsoft Entra ID

## ディレクトリ構成 📁

```text
gpt-image-1-streamlit/
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── static/
│   └── theme.css            # カスタムCSS
├── app.py                   # Streamlitアプリケーション本体
├── requirements.txt         # Python依存パッケージリスト
├── .env.example             # 環境変数テンプレート
├── Dockerfile               # Dockerイメージ定義
├── .dockerignore            # Docker除外設定
├── .gitignore               # Git除外設定
├── LICENSE                  # MITライセンス
└── README.md                # このファイル

# ローカル開発時に作成されるファイル (Git管理外)
├── .venv/                   # uv仮想環境
├── .env                     # ローカル用環境変数
└── .streamlit/secrets.toml  # Streamlitシークレット (任意)
```

## セットアップとローカル実行 🚀

### 前提条件

* Python 3.10+ (3.12推奨)
* uv (推奨: `brew install uv` または `curl -LsSf https://astral.sh/uv/install.sh | sh`)
* Docker Desktop
* Azure CLI (Azureデプロイを行う場合)
* OpenAI APIキー または Azure OpenAI Service のエンドポイント/APIキー/デプロイ名

### 手順

1. **リポジトリをクローン (まだの場合):**

    ```bash
    git clone https://github.com/naoki1213mj/gpt-image-1-streamlit.git
    cd gpt-image-1-streamlit
    ```

2. **仮想環境の作成と有効化 (uv推奨):**

    ```bash
    uv venv .venv --python 3.12
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```

3. **依存パッケージのインストール (uv推奨):**

    ```bash
    uv pip sync requirements.txt
    # または: uv pip install -r requirements.txt
    ```

4. **環境変数ファイルの設定:**
    * `.env.example` をコピーして `.env` ファイルを作成します。

        ```bash
        cp .env.example .env
        ```

    * `.env` ファイルを開き、使用するAPIキーやエンドポイントなどを設定します。

        ```dotenv:.env (設定例)
        # --- 通常のOpenAI API を使う場合 ---
        AI_HOST=OpenAI
        OPENAI_API_KEY=sk-...
        EASY_AUTH_ENABLED=false
        LOG_LEVEL=INFO

        # --- Azure OpenAI Service を使う場合 ---
        AI_HOST=AzureOpenAI
        AZURE_OPENAI_API_IMAGE_ENDPOINT=https://your-resource.openai.azure.com
        AZURE_OPENAI_API_IMAGE_MODEL=gpt-image-1
        AZURE_OPENAI_API_IMAGE_KEY=your-api-key
        AZURE_OPENAI_API_VERSION=2025-04-01-preview
        EASY_AUTH_ENABLED=false
        LOG_LEVEL=INFO
        ```

5. **Streamlitシークレットの設定 (任意: `st.login` をローカルで使う場合):**
    * ローカルで `st.login("microsoft")` を試す場合は、`.streamlit/secrets.toml` ファイルを作成し、Entra IDアプリ登録情報を記述します。

        ```toml:.streamlit/secrets.toml (例)
        [microsoft]
        clientId = "YOUR_ENTRA_APPLICATION_CLIENT_ID"
        tenantId = "YOUR_ENTRA_TENANT_ID"
        # clientSecret = "..." # 必要に応じて
        redirectUri = "http://localhost:8501"
        ```

6. **ローカルで実行:**

    ```bash
    streamlit run app.py
    ```

    ブラウザで `http://localhost:8501` にアクセスします。

## Dockerでの実行 🐳

1. **Dockerイメージをビルド:**

    ```bash
    docker build -t gptimage1-streamlit:latest .
    ```

    * **Apple Silicon (ARM64) 環境の場合:** Azure (通常AMD64) へのデプロイを考慮し、ターゲットプラットフォームを指定してビルドします。

        ```bash
        docker buildx build --platform linux/amd64 -t gptimage1-streamlit:latest-amd64 --load .
        ```

2. **Dockerコンテナを実行:**

    ```bash
    # 通常のビルドの場合
    docker run --rm -p 8501:8501 -v $(pwd)/.env:/app/.env gptimage1-streamlit:latest

    # AMD64ビルドの場合
    # docker run --rm -p 8501:8501 -v $(pwd)/.env:/app/.env gptimage1-streamlit:latest-amd64
    ```

    * `-v $(pwd)/.env:/app/.env` でローカルの `.env` ファイルをコンテナ内にマウントし、APIキーなどを渡しています。

    ブラウザで `http://localhost:8501` にアクセスします。

## Azureへのデプロイ ☁️

このアプリケーションは、Azure App Service for Containers または Azure Container Apps へのデプロイに対応しています。詳細な手順（ACRへのプッシュ、PaaSリソース作成、Entra ID Easy Auth設定、Key Vault + Managed Identityによるシークレット管理）については、以下のZenn記事を参照してください。

**📄 Zenn記事:** [GPT-image-1 × Streamlit で作る画像生成AIアプリ - Entra ID認証連携からAzureデプロイまで完全解説](https://zenn.dev/chips0711/articles/28eee04b8f2cfd)

### デプロイに必要な主なAzureリソース

* Azure Container Registry (ACR)
* Azure App Service Plan & App Service (for Containers) **または** Azure Container Apps Environment & Container App
* Azure Key Vault
* Microsoft Entra ID (アプリ登録)

## 設定項目 ⚙️

### 環境変数 (`.env` / Azure App Settings)

* `AI_HOST`: `"AzureOpenAI"` または `"OpenAI"`。使用するAPIサービスを指定。
* `EASY_AUTH_ENABLED`: `"true"` または `"false"`。Easy Auth（Azure App Service認証）の有効化。
* `LOG_LEVEL`: `"DEBUG"` / `"INFO"` / `"WARNING"` / `"ERROR"`。ログレベルの設定。

#### Azure OpenAI Service を使用する場合 (`AI_HOST=AzureOpenAI`)

* `AZURE_OPENAI_API_IMAGE_ENDPOINT`: Azure OpenAI Service のエンドポイントURL。
* `AZURE_OPENAI_API_IMAGE_MODEL`: Azure OpenAI Service でのGPT-image-1モデルのデプロイ名。
* `AZURE_OPENAI_API_IMAGE_KEY`: Azure OpenAI APIキー。
* `AZURE_OPENAI_API_VERSION`: 使用するAzure OpenAI APIバージョン。

#### 通常のOpenAI API を使用する場合 (`AI_HOST=OpenAI`)

* `OPENAI_API_KEY`: OpenAI APIキー。

### Streamlit シークレット (`.streamlit/secrets.toml`)

* ローカルで `st.login("microsoft")` を使用する場合に、`[microsoft]` セクションに `clientId`, `tenantId`, `redirectUri` などを設定します。

### Streamlit 設定 (`.streamlit/config.toml`)

* アプリのテーマ (`[theme]`) やサーバー設定 (`[server]`) をカスタマイズできます。詳細はStreamlitドキュメントを参照してください。

### カスタムスタイル (`static/theme.css`)

* アプリケーションのビジュアルデザインをカスタマイズするCSSファイルです。ボタンスタイル、ギャラリー表示、アニメーションなどの見た目を定義しています。

## コントリビューション

バグ報告、機能提案、改善提案などはGitHub Issuesへお願いします。プルリクエストも歓迎します！

## ライセンス 📜

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

## 参考資料 📚

* **Zenn記事:** [GPT-image-1 × Streamlit で作る画像生成AIアプリ - Entra ID認証連携からAzureデプロイまで完全解説](https://zenn.dev/chips0711/articles/28eee04b8f2cfd)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [OpenAI API Documentation](https://platform.openai.com/docs)
* [Azure OpenAI Service Documentation](https://learn.microsoft.com/ja-jp/azure/ai-services/openai/)
* [Azure Container Apps Documentation](https://learn.microsoft.com/ja-jp/azure/container-apps/)
* [Azure App Service Documentation](https://learn.microsoft.com/ja-jp/azure/app-service/)
* [uv Documentation](https://docs.astral.sh/uv/)
