"""Export dữ liệu THẬT từ FT-Transformer V2 (đã train, hyperparameter Optuna-tuned) để build
trang demo animation (pipeline training + inference) cho mentor — mọi số liệu trong JSON xuất
ra đều tính trực tiếp từ model thật, KHÔNG phải minh hoạ tay/số giả.

Cách làm: sau khi train xong FT-Transformer V2, chọn 1 đơn hàng thật trong test set, tự tính
lại TỪNG BƯỚC (tokenize → Q/K/V → attention → FFN → CLS → logit → xác suất) bằng đúng trọng số
đã học của model — có kiểm chứng bằng cách so khớp với output thật của model (sai số ~0). Thêm
1 bước training (forward → loss → backward → AdamW update) trên model MỚI khởi tạo (để có ví dụ
"bước đầu tiên" sạch, m=0/v=0) — tách riêng khỏi model đã train dùng cho phần inference.

    python3 Baseline/export_ft_transformer_demo_data.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, build_features, load_raw
from mlp_common import apply_categorical_encoders, encode_categoricals, fit_numeric_scaler, transform_numeric
from split import align_categories, time_split
from train import ART, RAW
from train_ft_transformer import NUMERIC_FEATURES, FTTransformer

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = os.path.join(ART, "ft_transformer_demo")
os.makedirs(OUT_DIR, exist_ok=True)
EPOCHS, PATIENCE = 40, 5
TEST_EXAMPLE_IDX = 0  # chọn đơn ĐẦU TIÊN trong test — không "chọn lọc" theo kết quả đẹp


def r(x, nd=6):
    """Round an array-like/tensor to a plain nested list, an toàn cho json.dump."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def main():
    print("1 · Nạp data + chia train/valid/test (post-dispatch, khoá 13/07)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")

    print("2 · Tiền xử lý (log1p+scale numeric, label-encode categorical)")
    scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
    X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
    X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
    cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

    bp = json.load(open(os.path.join(ART, "metrics_ft_transformer_v2.json")))["best_params"]
    d_token, n_heads, n_layers, d_ffn, dropout = (
        bp["d_token"], bp["n_heads"], bp["n_layers"], bp["d_ffn"], bp["dropout"])
    lr, weight_decay, batch_size = bp["lr"], bp["weight_decay"], bp["batch_size"]
    print(f"  V2 params: d_token={d_token} n_heads={n_heads} n_layers={n_layers} "
          f"d_ffn={d_ffn} dropout={dropout:.3f} lr={lr:.2e} batch={batch_size}")
    d_head = d_token // n_heads

    X_tr_num_t = torch.tensor(X_tr_num, dtype=torch.float32)
    X_tr_cat_t = torch.tensor(X_tr_cat, dtype=torch.long)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32)
    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32)
    X_va_cat_t = torch.tensor(X_va_cat, dtype=torch.long)
    X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32)
    X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long)

    print("\n3 · Train FT-Transformer V2 thật (dùng cho phần Inference demo)")
    model = FTTransformer(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                           d_token=d_token, n_layers=n_layers, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n_train = len(X_tr_num)
    best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(X_tr_num_t[idx], X_tr_cat_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).numpy()
        auc = roc_auc_score(y_va, p_va)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch, no_improve = epoch, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stopping ở epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  best epoch {best_epoch} · valid AUC {best_auc:.4f} · tổng {n_params:,} tham số")

    with torch.no_grad():
        p_te_all = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).numpy()
    m_test = evaluate(y_te.values, p_te_all, "test", verbose=False)
    print(f"  test ROC-AUC={m_test['roc_auc']:.4f} Cancel PR-AUC={m_test['pr_auc_cancel']:.4f}")

    # =====================================================================================
    # 4 · Đơn hàng thật để đi xuyên suốt demo — chọn ĐƠN ĐẦU TIÊN trong test (không chọn lọc)
    # =====================================================================================
    print(f"\n4 · Chọn đơn test #{TEST_EXAMPLE_IDX} làm ví dụ xuyên suốt demo")
    i = TEST_EXAMPLE_IDX
    raw_row = X_te.iloc[i]
    y_true = int(y_te.iloc[i])
    x_num_raw = {f: float(raw_row[f]) for f in NUMERIC_FEATURES}
    x_cat_raw = {f: str(raw_row[f]) for f in CATEGORICAL_FEATURES}
    x_num_processed = X_te_num[i]
    x_cat_processed = X_te_cat[i]

    tokenizer = model.tokenizer
    x_num_t = torch.tensor(x_num_processed, dtype=torch.float32).unsqueeze(0)
    x_cat_t = torch.tensor(x_cat_processed, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        tokens = tokenizer(x_num_t, x_cat_t)[0]  # (seq_len, d_token)
    seq_len = tokens.shape[0]
    token_names = ["[CLS]"] + list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    assert len(token_names) == seq_len

    print(f"  chuỗi {seq_len} token (1 CLS + {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical)")

    # =====================================================================================
    # 5 · Tự tính lại TỪNG BƯỚC qua 2 lớp encoder bằng ĐÚNG trọng số đã học — verify khớp model thật
    # =====================================================================================
    print("\n5 · Tự tính lại forward pass (Q/K/V, attention, FFN) — kiểm chứng khớp model thật")
    x_running = tokens.clone()
    layers_export = []
    with torch.no_grad():
        for li in range(n_layers):
            layer = model.encoder.layers[li]
            x_in = x_running.clone()
            x_norm1 = layer.norm1(x_running)

            Wq, Wk, Wv = layer.self_attn.in_proj_weight.chunk(3, dim=0)
            bq, bk, bv = layer.self_attn.in_proj_bias.chunk(3, dim=0)
            Q = x_norm1 @ Wq.T + bq
            K = x_norm1 @ Wk.T + bk
            V = x_norm1 @ Wv.T + bv

            def split_heads(t):
                return t.view(seq_len, n_heads, d_head).transpose(0, 1)  # (n_heads, seq_len, d_head)

            Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
            scores = torch.matmul(Qh, Kh.transpose(-1, -2)) / math.sqrt(d_head)
            attn_weights = torch.softmax(scores, dim=-1)  # (n_heads, seq_len, seq_len)
            head_out = torch.matmul(attn_weights, Vh)      # (n_heads, seq_len, d_head)
            concat = head_out.transpose(0, 1).contiguous().view(seq_len, d_token)
            attn_out = concat @ layer.self_attn.out_proj.weight.T + layer.self_attn.out_proj.bias
            x_after_attn = x_running + attn_out

            x_norm2 = layer.norm2(x_after_attn)
            ff_hidden = F.gelu(x_norm2 @ layer.linear1.weight.T + layer.linear1.bias)
            ff_out = ff_hidden @ layer.linear2.weight.T + layer.linear2.bias
            x_running = x_after_attn + ff_out

            layers_export.append({
                "layer_index": li,
                "input": r(x_in),
                "norm1_output": r(x_norm1),
                "Q": r(Q), "K": r(K), "V": r(V),
                "attention_weights_per_head": r(attn_weights),  # (n_heads, seq_len, seq_len)
                "head_output_per_head": r(head_out),            # (n_heads, seq_len, d_head)
                "concat_heads": r(concat),
                "attn_out_proj": r(attn_out),
                "after_residual1": r(x_after_attn),
                "norm2_output": r(x_norm2),
                "ffn_hidden_gelu": r(ff_hidden[:, :16]),  # d_ffn có thể lớn — chỉ xuất 16 chiều đầu để xem trước
                "ffn_hidden_dim": int(ff_hidden.shape[1]),
                "ffn_out": r(ff_out),
                "output": r(x_running),
            })

        # verify khớp model thật
        with torch.no_grad():
            real_encoder_out = model.encoder(tokens.unsqueeze(0))[0]
        max_diff = float((real_encoder_out - x_running).abs().max())
        print(f"  sai số so với model.encoder() thật: {max_diff:.2e} (phải ~0)")
        assert max_diff < 1e-4, "Tự tính lại KHÔNG khớp model thật — kiểm tra lại công thức"

        cls_final = x_running[0]
        head = model.head
        h1 = head[0](cls_final)          # LayerNorm
        h2 = torch.relu(h1)              # ReLU (dropout=identity ở eval)
        w_out = head[3].weight[0]        # (d_token,)
        b_out = head[3].bias[0]
        logit = (h2 * w_out).sum() + b_out
        prob = torch.sigmoid(logit)

        real_logit = model(x_num_t, x_cat_t)[0]
        print(f"  sai số logit so với model thật: {float((real_logit - logit).abs()):.2e} (phải ~0)")

    inference_export = {
        "example_index": i,
        "raw_numeric": x_num_raw,
        "raw_categorical": x_cat_raw,
        "y_true": y_true,
        "processed_numeric": {f: round(float(v), 6) for f, v in zip(NUMERIC_FEATURES, x_num_processed)},
        "processed_categorical_code": {f: int(v) for f, v in zip(CATEGORICAL_FEATURES, x_cat_processed)},
        "token_names": token_names,
        "tokens_after_tokenizer": r(tokens),
        "num_weight": r(tokenizer.num_weight),   # (n_numeric, d_token) — trọng số riêng từng feature
        "num_bias": r(tokenizer.num_bias),
        "layers": layers_export,
        "cls_final": r(cls_final),
        "head_layernorm_output": r(h1),
        "head_relu_output": r(h2),
        "head_linear_weight": r(w_out),
        "head_linear_bias": round(float(b_out), 6),
        "logit": round(float(logit), 6),
        "probability_cancel_class_is_0": round(1 - float(prob), 6),
        "probability_accept": round(float(prob), 6),
        "prediction_at_threshold_0.5": "accept" if float(prob) >= 0.5 else "cancel",
        "correct": (float(prob) >= 0.5) == (y_true == 1),
    }

    # =====================================================================================
    # 6 · 1 bước training THẬT (model mới khởi tạo — để có ví dụ "bước đầu" sạch: m=0, v=0)
    # =====================================================================================
    print("\n6 · Minh hoạ 1 bước training thật (forward → loss → backward → AdamW) trên model MỚI")
    torch.manual_seed(7)
    fresh_model = FTTransformer(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                                 d_token=d_token, n_layers=n_layers, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=lr, weight_decay=weight_decay)
    beta1, beta2, eps = fresh_opt.defaults["betas"][0], fresh_opt.defaults["betas"][1], fresh_opt.defaults["eps"]

    batch_n = 8
    xb_num, xb_cat, yb = X_tr_num_t[:batch_n], X_tr_cat_t[:batch_n], y_tr_t[:batch_n]
    fresh_model.train()
    logits_b = fresh_model(xb_num, xb_cat)
    probs_b = torch.sigmoid(logits_b)
    loss_b = loss_fn(logits_b, yb)

    w_before = fresh_model.head[3].weight[0, 0].item()
    b_before = fresh_model.head[3].bias[0].item()

    fresh_opt.zero_grad()
    loss_b.backward()
    grad_w = fresh_model.head[3].weight.grad[0, 0].item()
    grad_b = fresh_model.head[3].bias.grad[0].item()

    fresh_opt.step()
    state_w = fresh_opt.state[fresh_model.head[3].weight]
    m1 = state_w["exp_avg"][0, 0].item()
    v1 = state_w["exp_avg_sq"][0, 0].item()
    step_t = state_w["step"].item() if hasattr(state_w["step"], "item") else state_w["step"]
    m_hat = m1 / (1 - beta1 ** step_t)
    v_hat = v1 / (1 - beta2 ** step_t)
    w_after = fresh_model.head[3].weight[0, 0].item()

    training_export = {
        "batch_size_shown": batch_n,
        "y_true": [int(v) for v in yb.tolist()],
        "probabilities_accept": r(probs_b),
        "loss_bce": round(float(loss_b), 6),
        "focus_param": "head[3].weight[0,0]  (trọng số cuối cùng nối chiều 0 của CLS -> logit)",
        "w_before": round(w_before, 6),
        "gradient_dL_dw": round(grad_w, 6),
        "b_before": round(b_before, 6),
        "gradient_dL_db": round(grad_b, 6),
        "adamw_beta1": beta1, "adamw_beta2": beta2, "adamw_eps": eps,
        "adamw_lr": lr, "adamw_weight_decay": weight_decay,
        "exp_avg_m1_biased": round(m1, 8),
        "exp_avg_sq_v1_biased": round(v1, 8),
        "m_hat_bias_corrected": round(m_hat, 8),
        "v_hat_bias_corrected": round(v_hat, 8),
        "w_after": round(w_after, 6),
        "n_total_params": n_params,
    }

    # =====================================================================================
    # 7 · Ghi JSON
    # =====================================================================================
    payload = {
        "meta": {
            "model": "FT-Transformer V2 (Optuna-tuned)",
            "d_token": d_token, "n_heads": n_heads, "n_layers": n_layers, "d_ffn": d_ffn,
            "dropout": dropout, "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
            "n_numeric": len(NUMERIC_FEATURES), "n_categorical": len(CATEGORICAL_FEATURES),
            "seq_len": seq_len, "n_total_params": n_params,
            "test_roc_auc": round(m_test["roc_auc"], 4), "test_cancel_pr_auc": round(m_test["pr_auc_cancel"], 4),
            "best_epoch": best_epoch, "best_valid_auc": round(best_auc, 4),
        },
        "inference": inference_export,
        "training_step": training_export,
    }
    out_path = os.path.join(OUT_DIR, "ft_transformer_demo_data.json")
    json.dump(payload, open(out_path, "w"), ensure_ascii=False)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n✓ -> {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
