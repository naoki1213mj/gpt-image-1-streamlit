from __future__ import annotations

"""
GPT-image-1 Streamlit アプリケーション
========================================

このアプリケーションは、OpenAIのGPT-image-1モデルを使用して画像を生成・編集するための
Streamlitベースのウェブアプリケーションです。

主な機能:
- テキストから画像生成 (txt2img)
- 画像から画像編集 (img2img)
- Azure OpenAI Service と OpenAI API の両方に対応
- Microsoft Entra ID 認証 (Azure App Service Easy Auth) サポート

作者: naoki1213mj
ライセンス: MIT
"""

import base64
import io
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import requests
import streamlit as st
from openai import AzureOpenAI, OpenAI
from PIL import Image


# ─────────────────────────────────────────────────────────
# リトライ設定（バージョン差異対策）
# openaiライブラリのバージョンによってRetryConfigの場所が異なるため、
# 複数パターンを試行してインポートします
try:
    from openai import RetryConfig  # v1.17–1.75
except ImportError:
    try:
        from openai._types import RetryConfig  # v1.76+
    except ImportError:
        RetryConfig = None  # さらに古い／新しい場合はNoneで処理

# ─────────────────────────────────────────────────────────
# アプリケーション設定定数
# ─────────────────────────────────────────────────────────
IMAGE_LONG_EDGE_MAX = 2048  # 生成画像の最大長辺サイズ（ピクセル）
HISTORY_MAX = 30  # 履歴に保存する最大画像数
TIMEOUT_SEC = 300  # API呼び出しのタイムアウト（秒）


# ─────────────────────────────────────────────────────────
# Enumクラス：画像生成パラメータの定義
# ─────────────────────────────────────────────────────────

class ImageSize(str, Enum):
    """
    画像生成時のサイズオプション
    
    AUTO: プロンプトから自動判定
    SQUARE: 正方形 (1024x1024)
    LANDSCAPE: 横長 (1536x1024)
    PORTRAIT: 縦長 (1024x1536)
    """
    AUTO = "auto"
    SQUARE = "1024x1024"
    LANDSCAPE = "1536x1024"
    PORTRAIT = "1024x1536"


class ImageFormat(str, Enum):
    """
    画像の出力フォーマット
    
    PNG: 可逆圧縮、透過対応
    JPEG: 非可逆圧縮、ファイルサイズ小
    WEBP: 現代的なフォーマット、圧縮効率良
    """
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ImageQuality(str, Enum):
    """
    画像生成の品質レベル
    
    AUTO: 自動選択
    HIGH: 高品質（処理時間長）
    MEDIUM: 中品質
    LOW: 低品質（処理時間短）
    """
    AUTO = "auto"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Background(str, Enum):
    """
    背景の透過設定
    
    AUTO: 自動選択
    TRANSPARENT: 透明背景
    OPAQUE: 不透明背景
    """
    AUTO = "auto"
    TRANSPARENT = "transparent"
    OPAQUE = "opaque"


class Moderation(str, Enum):
    """
    コンテンツモデレーションの厳格度
    
    AUTO: 自動選択
    LOW: 低い（制限緩い）
    """
    AUTO = "auto"
    LOW = "low"


# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    """
    アプリケーション設定クラス
    
    環境変数から設定を読み込み、イミュータブルな設定オブジェクトを提供します。
    
    Attributes:
        azure_deployment: Azure OpenAI のデプロイ名
        azure_endpoint: Azure OpenAI のエンドポイントURL
        azure_key: Azure OpenAI の APIキー
        azure_api_version: Azure OpenAI の APIバージョン
        openai_key: OpenAI の APIキー（通常のOpenAI API使用時）
        log_level: ログレベル (DEBUG/INFO/WARNING/ERROR)
        easy_auth: Easy Auth（Azure認証）の有効化フラグ
    """
    azure_deployment: Optional[str]
    azure_endpoint: Optional[str]
    azure_key: Optional[str]
    azure_api_version: str
    openai_key: Optional[str]
    log_level: str
    easy_auth: bool

    @classmethod
    def from_env(cls) -> "Config":
        """
        環境変数から設定を読み込んでConfigインスタンスを作成
        
        .envファイルがあれば自動的に読み込まれます。
        
        Returns:
            Config: 設定オブジェクト
        """
        dotenv.load_dotenv(override=False)
        return cls(
            azure_deployment=os.getenv("AZURE_OPENAI_API_IMAGE_MODEL"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_IMAGE_ENDPOINT"),
            azure_key=os.getenv("AZURE_OPENAI_API_IMAGE_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            openai_key=os.getenv("OPENAI_API_KEY"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            easy_auth=os.getenv("EASY_AUTH_ENABLED", "false").lower() == "true",
        )


# ─────────────────────────────────────────────────────────
class Auth:
    """
    認証管理クラス
    
    Azure App Service Easy Auth または Streamlit標準の認証を管理します。
    """
    
    @staticmethod
    def _secrets_ready() -> bool:
        """
        Streamlitシークレット（認証設定）が利用可能かチェック
        
        Returns:
            bool: シークレットが正しく設定されている場合True
        """
        try:
            return bool(st.secrets["auth"]["microsoft"]["client_id"])
        except Exception:
            return False

    @staticmethod
    def ensure(cfg: Config) -> None:
        """
        認証が必要な場合に認証を強制
        
        Easy Authが有効な場合はスキップ（Azure側で認証済み）。
        ローカル実行時でシークレットが設定されている場合はMicrosoft認証を実行。
        
        Args:
            cfg: アプリケーション設定
        """
        if cfg.easy_auth:
            # Easy Auth有効時はAzure側で認証済みなのでスキップ
            return
        if not Auth._secrets_ready():
            # シークレット未設定時は警告のみ表示
            st.warning(".streamlit/secrets.toml が見つからないので認証をスキップします")
            return
        # 未ログインの場合はMicrosoft認証画面へ
        if not getattr(st.user, "is_logged_in", False):
            st.login("microsoft")
            st.stop()


# ─────────────────────────────────────────────────────────
class AIClient:
    """
    AI画像生成クライアント
    
    OpenAI API または Azure OpenAI Service API を使用して画像を生成・編集します。
    設定に応じて自動的に適切なクライアントを初期化します。
    
    Attributes:
        cfg: アプリケーション設定
        client: OpenAI または AzureOpenAI クライアントインスタンス
        model: 使用するモデル名
        is_azure: Azure OpenAI Service を使用しているかのフラグ
    """
    
    def __init__(self, cfg: Config) -> None:
        """
        AIクライアントの初期化
        
        環境変数の設定に基づいて、OpenAI API または Azure OpenAI Service の
        いずれかのクライアントを初期化します。
        
        Args:
            cfg: アプリケーション設定
            
        Raises:
            RuntimeError: OpenAI APIキーが設定されていない場合
        """
        logging.basicConfig(level=cfg.log_level.upper())
        self.cfg = cfg
        
        # タイムアウトとリトライ設定を共通化
        common: Dict[str, Any] = {"timeout": TIMEOUT_SEC}
        if RetryConfig:
            # リトライ設定: 最大5回、1-20秒の指数バックオフ
            common["retry_config"] = RetryConfig(max_retries=5, min_seconds=1, max_seconds=20)

        # Azure OpenAI Service または 通常のOpenAI API を選択
        if cfg.azure_endpoint and cfg.azure_key:
            # Azure OpenAI Service を使用
            self.client = AzureOpenAI(
                azure_endpoint=cfg.azure_endpoint,
                api_key=cfg.azure_key,
                api_version=cfg.azure_api_version,
                **common,
            )
            self.model = cfg.azure_deployment or "gpt-image-1"
            self.is_azure = True
        else:
            # 通常のOpenAI API を使用
            if not cfg.openai_key:
                raise RuntimeError("OPENAI_API_KEY が設定されていません")
            self.client = OpenAI(api_key=cfg.openai_key, **common)
            self.model = "gpt-image-1"
            self.is_azure = False

    @staticmethod
    def _cap(img: Image.Image) -> Image.Image:
        """
        画像サイズを制限
        
        長辺がIMAGE_LONG_EDGE_MAXを超える場合、アスペクト比を保ったまま
        サムネイル化して縮小します。
        
        Args:
            img: PIL Image オブジェクト
            
        Returns:
            Image.Image: サイズ調整後の画像
        """
        if max(img.size) > IMAGE_LONG_EDGE_MAX:
            img.thumbnail((IMAGE_LONG_EDGE_MAX, IMAGE_LONG_EDGE_MAX))
        return img

    def generate(
        self, *, prompt: str, background: str, moderation: str, compression: int, fmt: str, quality: str, size: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        テキストから画像を生成（txt2img）
        
        プロンプトに基づいて新しい画像を生成します。
        
        Args:
            prompt: 画像生成プロンプト
            background: 背景設定 (auto/transparent/opaque)
            moderation: モデレーション設定 (auto/low)
            compression: 圧縮率 (0-100、PNG以外で有効)
            fmt: 画像フォーマット (png/jpeg/webp)
            quality: 品質設定 (auto/high/medium/low)
            size: 画像サイズ (auto/1024x1024/1536x1024/1024x1536)
            
        Returns:
            Tuple[Image.Image, Dict[str, Any]]: 生成画像とメタデータのタプル
        """
        t0 = time.time()
        
        # OpenAI API を呼び出して画像生成
        rsp = self.client.images.generate(
            prompt=prompt,
            model=self.model,
            n=1,  # 生成枚数
            background=background,
            moderation=moderation,
            output_compression=compression,
            output_format=fmt,
            quality=quality,
            size=size,
        )
        
        # Base64エンコードされた画像をデコード＆サイズ調整
        img = self._cap(Image.open(io.BytesIO(base64.b64decode(rsp.data[0].b64_json))))
        
        # メタデータを構築
        meta = dict(
            prompt=prompt,
            settings=dict(
                background=background,
                moderation=moderation,
                compression=compression,
                format=fmt,
                quality=quality,
                size=size,
            ),
            generation_time=round(time.time() - t0, 2),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            type="text-to-image",
        )
        return img, meta

    def edit(self, image_bytes: bytes, *, prompt: str, quality: str, size: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        画像を編集（img2img）
        
        アップロードされた画像をプロンプトに基づいて編集します。
        Azure OpenAI Service の場合は直接REST APIを使用します。
        
        Args:
            image_bytes: 元画像のバイトデータ
            prompt: 編集指示プロンプト
            quality: 品質設定 (auto/high/medium/low)
            size: 画像サイズ (auto/1024x1024/1536x1024/1024x1536)
            
        Returns:
            Tuple[Image.Image, Dict[str, Any]]: 編集後の画像とメタデータのタプル
            
        Raises:
            RuntimeError: Azure API呼び出しが失敗した場合
        """
        t0 = time.time()
        
        if self.is_azure:
            # Azure OpenAI Service の場合は REST API を直接使用
            # （Python SDKがimg2imgに未対応のため）
            url = f"{self.cfg.azure_endpoint}/openai/deployments/{self.model}/images/edits?api-version={self.cfg.azure_api_version}"
            hdr = {"api-key": self.cfg.azure_key}
            data = {"prompt": prompt, "model": "gpt-image-1", "size": size, "n": 1, "quality": quality}
            files = {"image": ("image.png", image_bytes, "image/png")}
            
            rsp = requests.post(url, headers=hdr, data=data, files=files, timeout=TIMEOUT_SEC)
            if rsp.status_code != 200:
                raise RuntimeError(f"Azure edit 失敗: {rsp.status_code} {rsp.text}")
            b64 = rsp.json()["data"][0]["b64_json"]
        else:
            # 通常のOpenAI API の場合
            rsp = self.client.images.edit(
                model=self.model, image=image_bytes, prompt=prompt, n=1, quality=quality, size=size
            )
            b64 = rsp.data[0].b64_json
        
        # Base64エンコードされた画像をデコード＆サイズ調整
        img = self._cap(Image.open(io.BytesIO(base64.b64decode(b64))))
        
        # メタデータを構築
        meta = dict(
            prompt=prompt,
            settings=dict(quality=quality, size=size),
            generation_time=round(time.time() - t0, 2),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            type="image-to-image",
        )
        return img, meta


# ─────────────────────────────────────────────────────────
class UI:
    """
    ユーザーインターフェース管理クラス
    
    Streamlitを使用したUIコンポーネントの表示とユーザー操作を管理します。
    """
    
    @staticmethod
    def setup():
        """
        アプリケーションの基本設定とスタイルを適用
        
        ページ設定、カスタムCSSの読み込みを行います。
        """
        st.set_page_config("GPT-Image-1 Generator", "🖼️", layout="wide")
        css = "static/theme.css"
        if os.path.exists(css):
            st.markdown(f"<style>{open(css).read()}</style>", unsafe_allow_html=True)

    @staticmethod
    def header():
        """
        ページヘッダーの表示
        
        アプリケーションタイトルとユーザー情報（ログイン時）、
        ログアウトボタンを表示します。
        """
        c1, c2 = st.columns([9, 1])
        c1.markdown("## ✨ GPT-image-1 画像ジェネレーター")
        
        # ログイン済みの場合はユーザー名を表示
        if getattr(st, "user", None) and getattr(st.user, "name", None):
            c2.markdown(
                f"<span style='font-size:0.9rem;color:#94a3b8;'>👤 {st.user.name}</span>", unsafe_allow_html=True
            )
        if getattr(st.user, "is_logged_in", False):
            c2.button(
                "🔒 ログアウト", on_click=st.logout, key="logout_btn", help="アカウントを切り替える場合はこちらから"
            )
        st.divider()

    # ---------- チュートリアル ----------
    @staticmethod
    def _tutorial_body():
        """
        チュートリアル本文の表示
        
        初回訪問者向けの使い方ガイドを表示します。
        """
        st.markdown("### ようこそ！ 使い方ガイド")
        st.write("- **モード切替** で txt2img / img2img を選択")
        st.write("- サンプルボタンでプロンプトをワンタッチ挿入")
        st.write("- ⚙️ 詳細設定 で背景 / 品質 / サイズを調整")
        st.write("- 生成・編集した画像はギャラリーに自動保存")
        if st.button("はじめる", key="tut_close"):
            st.session_state["tutorial_shown"] = True
            st.rerun()

    @staticmethod
    def _show_expander_fallback():
        """
        フォールバック用チュートリアル表示（Expander使用）
        
        st.dialog が使用できない環境向けの代替表示方法
        """
        with st.expander("📝 初回チュートリアル", expanded=True):
            UI._tutorial_body()

    @staticmethod
    def tutorial_modal():
        """
        チュートリアルモーダルの表示
        
        Streamlitのバージョンに応じて、dialog、experimental_dialog、
        またはExpanderのいずれかを使用してチュートリアルを表示します。
        既に表示済みの場合はスキップします。
        """
        if st.session_state.get("tutorial_shown"):
            return

        # 1) experimental_dialog が存在するか？
        if hasattr(st, "experimental_dialog"):
            dlg_obj = st.experimental_dialog  # 1.25– は decorator / 1.34– は context manager
            # context-manager かどうか判定
            if callable(dlg_obj) and not hasattr(dlg_obj, "__enter__"):
                # decorator 形式のみ → フォールバック
                UI._show_expander_fallback()
            else:
                with st.experimental_dialog("📝 初回チュートリアル"):
                    UI._tutorial_body()
            return

        # 2) dialog が存在するか？
        if hasattr(st, "dialog"):
            dlg_obj = st.dialog
            if callable(dlg_obj) and not hasattr(dlg_obj, "__enter__"):
                # decorator 形式のみ
                UI._show_expander_fallback()
            else:
                with st.dialog("📝 初回チュートリアル"):
                    UI._tutorial_body()
            return

        # 3) どちらも無い → Expander
        UI._show_expander_fallback()

    @staticmethod
    def add_hist(img: Image.Image, meta: Dict[str, Any]):
        """
        生成画像を履歴に追加
        
        生成された画像とメタデータを履歴リストの先頭に追加します。
        履歴の最大数を超えた場合は古いものから削除します。
        
        Args:
            img: 生成された画像
            meta: 画像のメタデータ（プロンプト、設定、生成時間など）
        """
        hist = st.session_state.setdefault("history", [])
        hist.insert(0, dict(image=img, meta=meta, id=str(time.time())))
        if len(hist) > HISTORY_MAX:
            hist[:] = hist[:HISTORY_MAX]

    @staticmethod
    def gallery():
        """
        生成履歴ギャラリーの表示
        
        生成された画像を「すべて」「テキスト→画像」「画像→画像」の
        3つのカテゴリに分けて表示します。
        """
        hist = st.session_state.get("history", [])
        if not hist:
            return
        st.markdown("### 🖼️ 生成履歴")
        
        # 3つのカテゴリで表示
        for tag, title in [("all", "すべて"), ("txt", "テキスト→画像 (txt2img)"), ("img", "画像→画像 (img2img)")]:
            subset = (
                hist
                if tag == "all"
                else [h for h in hist if h["meta"]["type"].startswith("text" if tag == "txt" else "image")]
            )
            st.markdown(f"#### {title}")
            UI._gal_items(subset, tag)

    @staticmethod
    def _gal_items(items: List[Dict], tag: str):
        """
        ギャラリーアイテムの表示
        
        画像を3列のグリッドレイアウトで表示し、各画像に
        ダウンロードボタンを付けます。
        
        Args:
            items: 表示するアイテムのリスト
            tag: カテゴリタグ（キー生成用）
        """
        cols = st.columns(3)
        for i, it in enumerate(items):
            with cols[i % 3]:
                # PNG形式でバッファに保存
                buf = io.BytesIO()
                it["image"].save(buf, format="PNG")
                
                # 画像表示
                st.image(it["image"], use_container_width=True)
                
                # ダウンロードボタン
                st.download_button(
                    "💾 保存",
                    buf.getvalue(),
                    file_name=f"{it['id']}.png",
                    mime="image/png",
                    key=f"dl_{tag}_{it['id']}_{i}",
                    help="PNG形式でダウンロード",
                    use_container_width=True,
                )

    # ---- プロンプト＆スタイル例
    # プロンプト例の辞書：様々なシーンやテーマのプロンプトを用意
    PROMPTS = {
        "風景": "山頂から見る雄大な景色、朝日が雲海を照らす、雪をかぶった山々",
        "都市": "夜の東京、ネオンと雨、サイバーパンクスタイル",
        "ファンタジー": "ドラゴンが飛ぶ魔法の森、神秘的な光、幻想的な世界",
        "和風": "日本の伝統的な庭園、紅葉、苔むした石灯籠",
        "食べ物": "美味しそうな和食の定食、温かみのある照明",
        "アニメ": "ジブリ風のファンタジー世界、若い冒険者",
        "未来": "2150年の都市、空飛ぶ車、ホログラム広告",
        "ポートレート": "自然光で照らされた女性の肖像、背景ボケ",
    }
    
    # スタイル変換例の辞書：画像編集時のスタイル変換プロンプト
    STYLES = {
        "ジブリ": "ジブリ風のアニメスタイルに変換",
        "水彩画": "繊細な水彩画スタイルに変換",
        "油絵": "印象派の油絵スタイルに変換",
        "漫画": "日本の漫画風イラストに変換",
        "ネオン": "サイバーパンク風にネオンカラーで強調",
        "夕暮れ": "夕暮れの温かいオレンジ色に変更",
        "冬景色": "雪景色に変更",
        "ファンタジー": "魔法の世界風に変更",
    }

    @staticmethod
    def prompt_examples():
        """
        プロンプト例ボタンの表示
        
        事前定義されたプロンプト例をボタンとして表示し、
        クリックされるとテキストエリアにコピーします。
        """
        st.markdown("##### 📝 プロンプト例")
        cols = st.columns(4)
        for i, (k, v) in enumerate(UI.PROMPTS.items()):
            with cols[i % 4]:
                if st.button(k, key=f"prom_{k}", help="クリックでプロンプトにコピー"):
                    st.session_state["txt_prompt_tmp"] = v
                    st.rerun()

    @staticmethod
    def style_examples():
        """
        スタイル例ボタンの表示
        
        事前定義されたスタイル変換例をボタンとして表示し、
        クリックされると編集プロンプトにコピーします。
        """
        st.markdown("##### 🎨 スタイル例")
        cols = st.columns(4)
        for i, (k, v) in enumerate(UI.STYLES.items()):
            with cols[i % 4]:
                if st.button(k, key=f"style_{k}", help="クリックで編集プロンプトにコピー"):
                    st.session_state["img_prompt_tmp"] = v
                    st.rerun()


# ─────────────────────────────────────────────────────────
def main():
    """
    メインアプリケーション関数
    
    Streamlitアプリケーションのエントリーポイント。
    以下の処理を実行します：
    
    1. 設定の読み込み
    2. 認証の確認
    3. UIの初期化
    4. AIクライアントの初期化
    5. モード選択とそれに応じた処理
       - テキスト→画像生成 (txt2img)
       - 画像→画像編集 (img2img)
    6. 生成履歴ギャラリーの表示
    """
    # 環境変数から設定を読み込み
    cfg = Config.from_env()
    
    # 認証を確認（必要に応じてログイン画面へ）
    Auth.ensure(cfg)
    
    # UI初期化とAIクライアント作成
    UI.setup()
    ai = AIClient(cfg)
    
    # ヘッダーとチュートリアルを表示
    UI.header()
    UI.tutorial_modal()

    # モード選択（テキスト→画像 or 画像→画像）
    mode = st.radio(
        "操作モード",
        ["テキスト→画像(txt2img)", "画像→画像(img2img)"],
        horizontal=True,
        help="生成方法を選択してください",
    )
    st.divider()

    # ======== テキスト→画像（txt2img）モード ========
    if mode.startswith("テキスト"):
        # プロンプト入力エリア（例ボタンからのコピーをサポート）
        default_val = st.session_state.pop("txt_prompt_tmp", st.session_state.get("txt_prompt", ""))
        prompt = st.text_area(
            "✏️ プロンプト",
            key="txt_prompt",
            value=default_val,
            height=120,
            placeholder="例: A neon cyber-punk skyline at dusk, flying cars, holographic billboards…",
            help="生成したいイメージをできるだけ具体的に入力してください",
        )
        
        # プロンプト例ボタン表示
        UI.prompt_examples()

        # 詳細設定エリア（折りたたみ可能）
        with st.expander("⚙️ 詳細設定", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                bg = st.selectbox(
                    "背景", [e.value for e in Background], key="bg", help="背景の透明／不透明などを指定します"
                )
                mod = st.selectbox(
                    "モデレーション", [e.value for e in Moderation], key="mod", help="コンテンツフィルターの強さ"
                )
            with c2:
                qual = st.selectbox(
                    "品質", [e.value for e in ImageQuality], key="qual", help="高品質ほど処理時間が長くなります"
                )
                fmt = st.selectbox(
                    "フォーマット", [e.value for e in ImageFormat], key="fmt", help="透過が必要なら PNG 推奨"
                )
            comp = st.slider(
                "圧縮率 (JPEG/WebP)",
                0,
                100,
                100,
                key="comp",
                disabled=(fmt == "png"),
                help="数値を下げるほど高圧縮 (低画質) になります",
            )
            size = st.selectbox(
                "サイズ", [e.value for e in ImageSize], key="size", help="auto はプロンプト内容をもとに自動で判定"
            )

        # 画像生成ボタン
        if st.button("🚀 画像生成", type="primary", use_container_width=True, help="指定内容で画像を生成"):
            txt = st.session_state["txt_prompt"].strip()
            if not txt:
                st.warning("プロンプトを入力してください")
            else:
                # プログレスバー表示＆画像生成実行
                prog = st.progress(0, "生成中…")
                img, meta = ai.generate(
                    prompt=txt, background=bg, moderation=mod, compression=comp, fmt=fmt, quality=qual, size=size
                )
                prog.progress(100, "完了")
                
                # 生成結果を表示
                st.image(img, use_container_width=True, caption=f"{meta['generation_time']}秒")
                
                # 履歴に追加してお祝い表示
                UI.add_hist(img, meta)
                st.balloons()

    # ======== 画像→画像（img2img）モード ========
    else:
        c1, c2 = st.columns([1, 1])
        
        # 左カラム：入力と設定
        with c1:
            # 画像アップローダー
            up = st.file_uploader(
                "入力画像",
                type=["png", "jpg", "jpeg"],
                key="up",
                accept_multiple_files=False,
                help="編集したい元画像をアップロードしてください",
            )
            
            # 編集プロンプト入力（スタイル例からのコピーをサポート）
            default_val = st.session_state.pop("img_prompt_tmp", st.session_state.get("img_prompt", ""))
            pr = st.text_area(
                "✏️ 編集プロンプト",
                key="img_prompt",
                value=default_val,
                height=100,
                placeholder="例: ジブリ風の温かいアニメスタイルに変換…",
                help="↑ボタンでスタイル例をコピーできます",
            )
            
            # スタイル例ボタン表示
            UI.style_examples()
            
            # 詳細設定
            with st.expander("⚙️ 詳細設定", expanded=False):
                qual = st.selectbox("品質", [e.value for e in ImageQuality], key="e_qual", help="出力画像の品質")
                size = st.selectbox(
                    "サイズ", [e.value for e in ImageSize], key="e_size", help="元画像比で変形が起こる場合があります"
                )
            
            # 編集実行ボタン
            edit = st.button(
                "🖌️ 画像編集", type="primary", use_container_width=True, help="アップロード画像を指定スタイルで再生成"
            )

        # 右カラム：プレビューと情報
        with c2:
            if up:
                # アップロードされた画像を表示
                prev = Image.open(up)
                st.image(prev, use_container_width=True, caption="入力画像")
                
                # 画像情報を表示
                with st.expander("📊 画像情報", expanded=False):
                    w, h = prev.size
                    fmt0 = prev.format or "?"
                    st.write(f"**サイズ:** {w}×{h}")
                    st.write(f"**形式:** {fmt0}")
                    st.write(f"**容量:** {len(up.getvalue())/1024:.1f} KB")

        # 編集実行
        if edit:
            if not up:
                st.warning("画像をアップロードしてください")
                st.stop()
            txt = st.session_state["img_prompt"].strip()
            if not txt:
                st.warning("編集プロンプトを入力してください")
                st.stop()
            
            # プログレスバー表示＆画像編集実行
            prog = st.progress(0, "編集中…")
            img, meta = ai.edit(up.getvalue(), prompt=txt, quality=qual, size=size)
            prog.progress(100, "完了")
            
            # 編集前後を並べて表示
            a, b = st.columns(2)
            a.image(prev, use_container_width=True, caption="編集前")
            b.image(img, use_container_width=True, caption=f"編集後 ({meta['generation_time']}秒)")
            
            # 履歴に追加してお祝い表示
            UI.add_hist(img, meta)
            st.balloons()

    # 生成履歴ギャラリーを表示
    UI.gallery()


# ─────────────────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
