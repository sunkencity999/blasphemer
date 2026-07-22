"""Unit tests for the optimization report exporter."""

import csv

from heretic.reporting import (
    TrialRecord,
    compute_pareto_front,
    export_report,
)


def _record(index, kl, refusals, total=100, params=None, direction=None):
    return TrialRecord(
        index=index,
        kl_divergence=kl,
        refusals=refusals,
        total_prompts=total,
        direction_index=direction,
        parameters=params or {},
    )


class TestTrialRecord:
    def test_refusal_rate(self):
        assert _record(1, 0.1, 5, total=100).refusal_rate == 5.0

    def test_refusal_rate_zero_total_is_safe(self):
        assert _record(1, 0.1, 0, total=0).refusal_rate == 0.0

    def test_flat_parameters_per_layer(self):
        rec = _record(1, 0.1, 0, direction=None, params={"attn.o_proj": {"max_weight": 1.2}})
        flat = rec.flat_parameters()
        assert flat["direction_index"] == "per layer"
        assert flat["attn.o_proj.max_weight"] == "1.2000"

    def test_flat_parameters_global_direction(self):
        rec = _record(1, 0.1, 0, direction=12.5)
        assert rec.flat_parameters()["direction_index"] == "12.5000"


class TestParetoFront:
    def test_dominated_trials_excluded(self):
        records = [
            _record(1, 0.1, 5),   # Pareto
            _record(2, 0.2, 2),   # Pareto
            _record(3, 0.3, 8),   # dominated by 1
            _record(4, 0.15, 5),  # dominated by 1 (same refusals, worse KL)
        ]
        front = compute_pareto_front(records)
        indices = {r.index for r in front}
        assert indices == {1, 2}

    def test_front_sorted_by_refusals_then_kl(self):
        records = [_record(1, 0.1, 5), _record(2, 0.2, 2)]
        front = compute_pareto_front(records)
        assert [r.index for r in front] == [2, 1]

    def test_empty_input(self):
        assert compute_pareto_front([]) == []


class TestExportReport:
    def test_writes_csv_and_markdown(self, tmp_path):
        records = [
            _record(1, 0.10, 5, params={"attn.o_proj": {"max_weight": 1.1}}),
            _record(2, 0.20, 2, params={"attn.o_proj": {"max_weight": 1.3}}),
            _record(3, 0.30, 8, params={"attn.o_proj": {"max_weight": 0.9}}),
        ]
        out = export_report(
            records,
            model_name="test/model",
            output_dir=tmp_path / "report",
            base_refusals=50,
            total_prompts=100,
            elapsed_seconds=125.0,
        )

        csv_path = out / "trials.csv"
        md_path = out / "report.md"
        assert csv_path.exists()
        assert md_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        # Rows are sorted by index; row 3 is dominated, rows 1 & 2 are Pareto.
        by_index = {r["index"]: r for r in rows}
        assert by_index["1"]["pareto_optimal"] == "True"
        assert by_index["2"]["pareto_optimal"] == "True"
        assert by_index["3"]["pareto_optimal"] == "False"
        assert "attn.o_proj.max_weight" in rows[0]

        md = md_path.read_text(encoding="utf-8")
        assert "test/model" in md
        assert "Pareto-optimal trials" in md
        assert "Recommended trial" in md
        # Fewest-refusals Pareto trial (#2) is the recommendation.
        assert "Trial **2**" in md

    def test_empty_records_produces_valid_files(self, tmp_path):
        out = export_report([], model_name="m", output_dir=tmp_path / "r")
        assert (out / "trials.csv").exists()
        assert (out / "report.md").exists()
