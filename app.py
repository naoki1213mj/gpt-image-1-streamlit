from __future__ import annotations

# GPT-image-1 Streamlit – Easy Auth＆ローカル両対応（txt2img / img2img）
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
import dotenv
import requests
import streamlit as st
from openai import AzureOpenAI, OpenAI
from PIL import Image

# GPT-image-1 Streamlit – Easy Auth＆ローカル両対応（txt2img / img2img）


# ─────────────────────────────────────────────────────────
# リトライ設定（バージョン差異対策）
try:
    from openai import RetryConfig  # v1.17–1.75
except ImportError:
    try:
        from openai._types import RetryConfig  # v1.76+
    except ImportError:
        RetryConfig = None  # さらに古い／新しい場合

# ─────────────────────────────────────────────────────────
IMAGE_LONG_EDGE_MAX = 2048
HISTORY_MAX = 30
TIMEOUT_SEC = 300  # httpx / requests タイムアウト


# ─────────────────────────────────────────────────────────
class ImageSize(str, Enum):
    AUTO = "auto"
    SQUARE = "1024x1024"
    LANDSCAPE = "1536x1024"
    PORTRAIT = "1024x1536"


class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ImageQuality(str, Enum):
    AUTO = "auto"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Background(str, Enum):
    AUTO = "auto"
    TRANSPARENT = "transparent"
    OPAQUE = "opaque"


class Moderation(str, Enum):
    AUTO = "auto"
    LOW = "low"


# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    azure_deployment: Optional[str]
    azure_endpoint: Optional[str]
    azure_key: Optional[str]
    azure_api_version: str
    openai_key: Optional[str]
    log_level: str
    easy_auth: bool

    @classmethod
    def from_env(cls) -> "Config":
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
    @staticmethod
    def _secrets_ready() -> bool:
        try:
            return bool(st.secrets["auth"]["microsoft"]["client_id"])
        except Exception:
            return False

    @staticmethod
    def ensure(cfg: Config) -> None:
        if cfg.easy_auth:
            return
        if not Auth._secrets_ready():
            st.warning(".streamlit/secrets.toml が見つからないので認証をスキップします")
            return
        if not getattr(st.user, "is_logged_in", False):
            st.login("microsoft")
            st.stop()


# ─────────────────────────────────────────────────────────
class AIClient:
    def __init__(self, cfg: Config) -> None:
        logging.basicConfig(level=cfg.log_level.upper())
        self.cfg = cfg
        common: Dict[str, Any] = {"timeout": TIMEOUT_SEC}
        if RetryConfig:
            common["retry_config"] = RetryConfig(max_retries=5, min_seconds=1, max_seconds=20)

        if cfg.azure_endpoint and cfg.azure_key:
            self.client = AzureOpenAI(
                azure_endpoint=cfg.azure_endpoint,
                api_key=cfg.azure_key,
                api_version=cfg.azure_api_version,
                **common,
            )
            self.model = cfg.azure_deployment or "gpt-image-1"
            self.is_azure = True
        else:
            if not cfg.openai_key:
                raise RuntimeError("OPENAI_API_KEY が設定されていません")
            self.client = OpenAI(api_key=cfg.openai_key, **common)
            self.model = "gpt-image-1"
            self.is_azure = False

    # ---- サイズ制限
    @staticmethod
    def _cap(img: Image.Image) -> Image.Image:
        if max(img.size) > IMAGE_LONG_EDGE_MAX:
            img.thumbnail((IMAGE_LONG_EDGE_MAX, IMAGE_LONG_EDGE_MAX))
        return img

    # ---- txt2img
    def generate(
        self, *, prompt: str, background: str, moderation: str, compression: int, fmt: str, quality: str, size: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        t0 = time.time()
        rsp = self.client.images.generate(
            prompt=prompt,
            model=self.model,
            n=1,
            background=background,
            moderation=moderation,
            output_compression=compression,
            output_format=fmt,
            quality=quality,
            size=size,
        )
        img = self._cap(Image.open(io.BytesIO(base64.b64decode(rsp.data[0].b64_json))))
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

    # ---- img2img
    def edit(self, image_bytes: bytes, *, prompt: str, quality: str, size: str) -> Tuple[Image.Image, Dict[str, Any]]:
        t0 = time.time()
        if self.is_azure:
            url = f"{self.cfg.azure_endpoint}/openai/deployments/{self.model}/images/edits?api-version={self.cfg.azure_api_version}"
            hdr = {"api-key": self.cfg.azure_key}
            data = {"prompt": prompt, "model": "gpt-image-1", "size": size, "n": 1, "quality": quality}
            files = {"image": ("image.png", image_bytes, "image/png")}
            rsp = requests.post(url, headers=hdr, data=data, files=files, timeout=TIMEOUT_SEC)
            if rsp.status_code != 200:
                raise RuntimeError(f"Azure edit 失敗: {rsp.status_code} {rsp.text}")
            b64 = rsp.json()["data"][0]["b64_json"]
        else:
            rsp = self.client.images.edit(
                model=self.model, image=image_bytes, prompt=prompt, n=1, quality=quality, size=size
            )
            b64 = rsp.data[0].b64_json
        img = self._cap(Image.open(io.BytesIO(base64.b64decode(b64))))
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
    @staticmethod
    def setup():
        st.set_page_config("GPT-Image-1 Generator", "🖼️", layout="wide")
        css = "static/theme.css"
        if os.path.exists(css):
            st.markdown(f"<style>{open(css).read()}</style>", unsafe_allow_html=True)

    @staticmethod
    def header():
        c1, c2 = st.columns([9, 1])
        c1.markdown("## ✨ GPT-image-1 画像ジェネレーター")
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
        """最終フォールバック（Expander）"""
        with st.expander("📝 初回チュートリアル", expanded=True):
            UI._tutorial_body()

    @staticmethod
    def tutorial_modal():
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
        hist = st.session_state.setdefault("history", [])
        hist.insert(0, dict(image=img, meta=meta, id=str(time.time())))
        if len(hist) > HISTORY_MAX:
            hist[:] = hist[:HISTORY_MAX]

    @staticmethod
    def gallery():
        hist = st.session_state.get("history", [])
        if not hist:
            return
        st.markdown("### 🖼️ 生成履歴")
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
        cols = st.columns(3)
        for i, it in enumerate(items):
            with cols[i % 3]:
                buf = io.BytesIO()
                it["image"].save(buf, format="PNG")
                st.image(it["image"], use_container_width=True)
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
        st.markdown("##### 📝 プロンプト例")
        cols = st.columns(4)
        for i, (k, v) in enumerate(UI.PROMPTS.items()):
            with cols[i % 4]:
                if st.button(k, key=f"prom_{k}", help="クリックでプロンプトにコピー"):
                    st.session_state["txt_prompt_tmp"] = v
                    st.rerun()

    @staticmethod
    def style_examples():
        st.markdown("##### 🎨 スタイル例")
        cols = st.columns(4)
        for i, (k, v) in enumerate(UI.STYLES.items()):
            with cols[i % 4]:
                if st.button(k, key=f"style_{k}", help="クリックで編集プロンプトにコピー"):
                    st.session_state["img_prompt_tmp"] = v
                    st.rerun()


# ─────────────────────────────────────────────────────────
def main():
    cfg = Config.from_env()
    Auth.ensure(cfg)
    UI.setup()
    ai = AIClient(cfg)
    UI.header()
    UI.tutorial_modal()

    mode = st.radio(
        "操作モード",
        ["テキスト→画像(txt2img)", "画像→画像(img2img)"],
        horizontal=True,
        help="生成方法を選択してください",
    )
    st.divider()

    # ======== テキスト→画像 ========
    if mode.startswith("テキスト"):
        default_val = st.session_state.pop("txt_prompt_tmp", st.session_state.get("txt_prompt", ""))
        prompt = st.text_area(
            "✏️ プロンプト",
            key="txt_prompt",
            value=default_val,
            height=120,
            placeholder="例: A neon cyber-punk skyline at dusk, flying cars, holographic billboards…",
            help="生成したいイメージをできるだけ具体的に入力してください",
        )
        UI.prompt_examples()

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

        if st.button("🚀 画像生成", type="primary", use_container_width=True, help="指定内容で画像を生成"):
            txt = st.session_state["txt_prompt"].strip()
            if not txt:
                st.warning("プロンプトを入力してください")
            else:
                prog = st.progress(0, "生成中…")
                img, meta = ai.generate(
                    prompt=txt, background=bg, moderation=mod, compression=comp, fmt=fmt, quality=qual, size=size
                )
                prog.progress(100, "完了")
                st.image(img, use_container_width=True, caption=f"{meta['generation_time']}秒")
                UI.add_hist(img, meta)
                st.balloons()

    # ======== 画像→画像 ========
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            up = st.file_uploader(
                "入力画像",
                type=["png", "jpg", "jpeg"],
                key="up",
                accept_multiple_files=False,
                help="編集したい元画像をアップロードしてください",
            )
            default_val = st.session_state.pop("img_prompt_tmp", st.session_state.get("img_prompt", ""))
            pr = st.text_area(
                "✏️ 編集プロンプト",
                key="img_prompt",
                value=default_val,
                height=100,
                placeholder="例: ジブリ風の温かいアニメスタイルに変換…",
                help="↑ボタンでスタイル例をコピーできます",
            )
            UI.style_examples()
            with st.expander("⚙️ 詳細設定", expanded=False):
                qual = st.selectbox("品質", [e.value for e in ImageQuality], key="e_qual", help="出力画像の品質")
                size = st.selectbox(
                    "サイズ", [e.value for e in ImageSize], key="e_size", help="元画像比で変形が起こる場合があります"
                )
            edit = st.button(
                "🖌️ 画像編集", type="primary", use_container_width=True, help="アップロード画像を指定スタイルで再生成"
            )

        with c2:
            if up:
                prev = Image.open(up)
                st.image(prev, use_container_width=True, caption="入力画像")
                with st.expander("📊 画像情報", expanded=False):
                    w, h = prev.size
                    fmt0 = prev.format or "?"
                    st.write(f"**サイズ:** {w}×{h}")
                    st.write(f"**形式:** {fmt0}")
                    st.write(f"**容量:** {len(up.getvalue())/1024:.1f} KB")

        if edit:
            if not up:
                st.warning("画像をアップロードしてください")
                st.stop()
            txt = st.session_state["img_prompt"].strip()
            if not txt:
                st.warning("編集プロンプトを入力してください")
                st.stop()
            prog = st.progress(0, "編集中…")
            img, meta = ai.edit(up.getvalue(), prompt=txt, quality=qual, size=size)
            prog.progress(100, "完了")
            a, b = st.columns(2)
            a.image(prev, use_container_width=True, caption="編集前")
            b.image(img, use_container_width=True, caption=f"編集後 ({meta['generation_time']}秒)")
            UI.add_hist(img, meta)
            st.balloons()

    UI.gallery()


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
