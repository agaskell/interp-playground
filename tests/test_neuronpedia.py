from interp_playground.neuronpedia import example_text


def test_example_text_cleans_tokens():
    ex = {
        "max": 50.0,
        "tokens": ["<bos>", "▁We", "▁drove", "▁across", "▁the", "▁old", "▁bridge", ".", "\n", "▁Next"],
        "peak": 6,
    }
    assert example_text(ex) == "We drove across the old bridge. Next"


def test_example_text_window_is_before_and_after_peak():
    toks = [f"▁t{i}" for i in range(30)]
    ex = {"max": 1.0, "tokens": toks, "peak": 20}
    text = example_text(ex, before=2, after=1)
    assert text == "t18 t19 t20 t21"
