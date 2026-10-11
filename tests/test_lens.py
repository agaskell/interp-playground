import torch

from interp_playground.lens import lens_rows


def test_lens_rows_reports_top_tokens_and_final_token_prob():
    # 2 layers x vocab 4. Final prediction is token 3 (argmax of last layer).
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.05, 0.05, 0.1, 0.8]])
    rows = lens_rows(probs, layers=[5, 25], decode=lambda i: f"t{i}", top_k=2)
    assert rows == [
        {"layer": 5, "top": [["t0", 0.4], ["t1", 0.3]], "p_final": 0.1},
        {"layer": 25, "top": [["t3", 0.8], ["t2", 0.1]], "p_final": 0.8},
    ]
