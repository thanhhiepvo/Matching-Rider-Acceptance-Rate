"""Vẽ sơ đồ pipeline (preprocessing + model) trực tiếp trong code lúc build model, dùng
`sklearn.utils.estimator_html_repr` — lưu ra `artifacts/pipeline_<model>.html`, mở trực tiếp
bằng trình duyệt hoặc nhúng vào notebook (`display(HTML(open(path).read()))`).

LightGBM/XGBoost train qua native Booster API (`lgb.train`/`xgb.train`) để early-stop đúng
cách — không phải `Pipeline.fit()`. Sơ đồ cho 2 model này dựng 1 estimator sklearn TƯƠNG ĐƯƠNG
(`LGBMClassifier`/`XGBClassifier`, CÙNG hyperparameter) chỉ để trực quan hoá cấu trúc, KHÔNG
dùng để train/predict thật (xem train.py/train_xgboost.py — model thật vẫn là Booster).
CatBoost đã dùng `CatBoostClassifier` (sklearn-compatible) sẵn nên sơ đồ vẽ đúng object THẬT
đang được train/predict trong train_catboost.py.

    from pipeline_diagram import save_pipeline_diagram
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr


def save_pipeline_diagram(pipeline: Pipeline, path: str) -> None:
    """Ghi sơ đồ HTML của 1 sklearn Pipeline/estimator ra `path`. Không cần fit trước — sơ đồ
    thể hiện CẤU TRÚC (các bước + hyperparameter), không phải trạng thái đã fit."""
    html = estimator_html_repr(pipeline)
    with open(path, "w") as f:
        f.write(html)
    print(f"  sơ đồ pipeline -> {path}")


# FT-Transformer/Mamba (train_ft_transformer.py/train_mamba.py) là nn.Module (PyTorch) — không
# tự sklearn-compatible như CatBoostClassifier/LGBMClassifier/XGBClassifier nên KHÔNG vẽ được
# trực tiếp bằng estimator_html_repr. 4 class BaseEstimator "vỏ" dưới đây bọc lại đúng 3 khối
# kiến trúc thật (FeatureTokenizer -> khối mix [self-attention hoặc SSM] -> classification head)
# CÙNG hyperparameter đang dùng để train, chỉ để trực quan hoá — không transform/fit thật.
class _VizStep(BaseEstimator):
    """`fit()` no-op — chỉ để thoả điều kiện `hasattr(estimator, "fit")` mà sklearn 1.9's
    `estimator_html_repr` đòi hỏi (qua `check_is_fitted`) khi render sơ đồ, không dùng để
    transform/predict dữ liệu thật."""

    def fit(self, X=None, y=None):
        return self


class FeatureTokenizerViz(_VizStep):
    def __init__(self, n_numeric=17, cat_cardinalities=(3, 3, 15), d_token=32, cls_position="first"):
        self.n_numeric = n_numeric
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.cls_position = cls_position


class TransformerEncoderViz(_VizStep):
    """FT-Transformer — khối mix self-attention (nn.TransformerEncoder)."""

    def __init__(self, n_layers=3, n_heads=4, d_ffn=64, dropout=0.15):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.dropout = dropout


class MambaEncoderViz(_VizStep):
    """Mamba — khối mix selective state-space (S6 selective scan)."""

    def __init__(self, n_layers=3, d_state=16, d_conv=4, expand=2, dropout=0.15):
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout


class ClassificationHeadViz(_VizStep):
    def __init__(self, d_token=32, layers="LayerNorm -> ReLU -> Dropout -> Linear(1)"):
        self.d_token = d_token
        self.layers = layers
