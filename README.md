# GPT-image-1 生成アプリ（Streamlit + Azure OpenAI）

![screenshot](docs/screenshot.png) <!-- 任意: 画面キャプチャを置いたらパスだけ合わせる -->

> **テキスト → 画像 (txt2img)** と **画像 → 画像 (img2img)** の両モードを備えた  
> Microsoft Entra ID 認証対応の Streamlit アプリケーションです。  
> App Service（Easy Auth）にも Docker コンテナ（ACI / Container Apps など）にもデプロイできます。

---

## 📑 主な機能

| 機能 | 概要 |
|------|------|
| **Azure / OpenAI 両対応** | 環境変数で自動判定 |
| **Entra ID 認証** | ローカル→`st.login("microsoft")` ｜本番→Easy Auth 任せ |
| **txt2img / img2img** | モード切替ワンタッチ |
| **履歴ギャラリー** | 生成画像をセッション内でサムネイル保存 & ダウンロード |
| **チュートリアル Modal** | 初回訪問者にガイド表示（Streamlit ≥ 1.25） |

---

## 🗂️ ディレクトリ構成（抜粋）

```text
.
├─ app.py                 # アプリ本体
├─ .env.example           # 環境変数の雛形（鍵は入れない）
├─ static/
│  └─ theme.css           # ダークテーマ + カスタム CSS
├─ Dockerfile             # コンテナ用
├─ .streamlit/
│  └─ secrets.toml        # 開発時の Microsoft 認証情報
└─ README.md
