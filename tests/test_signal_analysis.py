from research.fineweb.analyze_signal_screen import compare


def test_cost_comparison_includes_construction_and_uses_observed_points():
    baseline = {"arm": "fresh", "data_order": 0, "status": "complete", "final_loss": 4.0,
                "train_seconds": 40.0, "total_seconds_excluding_teacher_training": 40.0,
                "construction": {}, "backward_calls": 64, "unique_source_tokens": 100,
                "cached_gradient_bytes": 0,
                "curve": [{"step": 0, "loss": 5.0, "train_seconds": 0.0},
                          {"step": 16, "loss": 4.7, "train_seconds": 10.0},
                          {"step": 32, "loss": 4.4, "train_seconds": 20.0},
                          {"step": 64, "loss": 4.0, "train_seconds": 40.0}]}
    teacher = dict(baseline, arm="teacher_top8", final_loss=3.9, train_seconds=8.0,
                   total_seconds_excluding_teacher_training=23.0,
                   construction={"construction_seconds": 15.0, "teacher_training_seconds_excluded_from_construction": 1000.0,
                                 "artifact_bytes": 300})
    output = compare({"results": [baseline, teacher], "initial_dev_loss": 5.0, "limitations": []})["rows"][1]
    assert output["baseline_point_at_or_before_candidate_cost"]["step"] == 32
    assert output["baseline_point_at_or_after_candidate_cost"]["step"] == 64
    assert output["teacher_training_seconds_additional"] == 1000.0
    assert output["payload_bytes"] == 300
