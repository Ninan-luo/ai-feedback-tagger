import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from demo import DemoClient
from user_text_encoder_v3 import (
    DataLoader, EncoderConfig, EncodingPipeline, MultiLevelLabelLearner,
    MultiLevelTextEncoder,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        content = next(self.responses)
        if isinstance(content, Exception):
            raise content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.silence = contextlib.redirect_stdout(self.output)
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = EncoderConfig(
            model="test-fixture", hierarchy_file=str(EXAMPLES / "hierarchy.csv"),
            max_retries=2, retry_delay=0, batch_size=100,
        )
        self.learner = MultiLevelLabelLearner(self.config)
        self.learner.learn_from_data(pd.read_csv(EXAMPLES / "train.csv"))
        self.learner.load_hierarchy_from_file(self.config.hierarchy_file)

    def test_hierarchy_maps_multiple_parents_and_deduplicates(self):
        client = FakeClient([{"二级编码": ["页面加载慢", "客服响应慢", "页面加载慢"], "confidence": "high"}])
        result = MultiLevelTextEncoder(self.config, self.learner, client).encode_single("页面和客服都很慢")
        self.assertEqual(result["一级编码"], ["产品体验", "服务体验"])
        self.assertEqual(result["二级编码"], ["页面加载慢", "客服响应慢"])
        self.assertEqual(client.calls, 1)

    def test_invalid_responses_retry_and_fail(self):
        for bad in ["not JSON", [], {"二级编码": None}, {"二级编码": ["不存在"], "confidence": "high"}, {"二级编码": [], "confidence": 0.9}]:
            with self.subTest(response=bad):
                client = FakeClient([bad, bad])
                result = MultiLevelTextEncoder(self.config, self.learner, client).encode_single("内容")
                self.assertEqual(client.calls, 2)
                self.assertEqual(result["二级编码"], ["编码失败"])
                self.assertEqual(result["confidence"], "none")

    def test_transient_failure_then_valid_response(self):
        client = FakeClient([RuntimeError("temporary"), {"二级编码": ["优惠不足"], "confidence": "low"}])
        result = MultiLevelTextEncoder(self.config, self.learner, client).encode_single("希望有优惠")
        self.assertEqual(result["一级编码"], ["价格与权益"])
        self.assertEqual(client.calls, 2)

    def test_empty_text_never_calls_api(self):
        client = FakeClient([])
        encoder = MultiLevelTextEncoder(self.config, self.learner, client)
        for text in [None, float("nan"), "", "   ", pd.NA]:
            self.assertEqual(encoder.encode_single(text)["confidence"], "none")
        self.assertEqual(client.calls, 0)

    def test_explicit_hierarchy_replaces_training_labels(self):
        small = Path(self.tmp.name) / "hierarchy.csv"
        small.write_text("一级编码,二级编码\n价格与权益,优惠不足\n")
        self.learner.load_hierarchy_from_file(str(small))
        self.assertEqual(self.learner.level_labels["二级编码"], ["优惠不足"])
        self.assertEqual(self.learner.reverse_hierarchy, {"优惠不足": "价格与权益"})

    def test_ambiguous_hierarchy_rejected(self):
        path = Path(self.tmp.name) / "ambiguous.csv"
        path.write_text("一级编码,二级编码\n产品体验,页面加载慢\n服务体验,页面加载慢\n")
        with self.assertRaisesRegex(ValueError, "多个一级"):
            self.learner.load_hierarchy_from_file(str(path))

    def test_example_budget_applies_to_both_prompt_modes(self):
        self.config.max_examples_total = 2
        encoder = MultiLevelTextEncoder(self.config, self.learner, FakeClient([]))
        for explicit in [True, False]:
            self.learner.hierarchy_from_file = explicit
            prompt = encoder._build_system_prompt()
            self.assertEqual(sum(line.startswith("  1.") or line.startswith("  2.") for line in prompt.splitlines()), 2)

    def test_pipeline_csv_preserves_rows_and_flags_review(self):
        path = Path(self.tmp.name) / "result.csv"
        result = EncodingPipeline(self.config, DemoClient()).run(
            str(EXAMPLES / "train.csv"), str(EXAMPLES / "predict.csv"), str(path)
        )
        saved = pd.read_csv(path, keep_default_na=False)
        self.assertEqual(len(saved), 6)
        self.assertEqual(saved["样本ID"].tolist(), [f"DEMO00{i}" for i in range(1, 7)])
        self.assertEqual(result["需人工复核"].tolist(), [False, False, True, False, True, True])
        self.assertEqual(saved.loc[5, "预测_二级编码"], "")

    def test_pipeline_xlsx_roundtrip(self):
        path = Path(self.tmp.name) / "result.xlsx"
        EncodingPipeline(self.config, DemoClient()).run(
            str(EXAMPLES / "train.csv"), str(EXAMPLES / "predict.csv"), str(path)
        )
        saved = DataLoader.load_excel(str(path))
        self.assertEqual(len(saved), 6)
        self.assertEqual(saved.loc[0, "预测_一级编码"], "产品体验、服务体验")

    def test_output_paths_cannot_overwrite_inputs_or_results(self):
        pipeline = EncodingPipeline(self.config, DemoClient())
        for target in [EXAMPLES / "train.csv", EXAMPLES / "predict.csv", EXAMPLES / "hierarchy.csv"]:
            with self.assertRaisesRegex(ValueError, "输入文件"):
                pipeline.run(str(EXAMPLES / "train.csv"), str(EXAMPLES / "predict.csv"), str(target))
        target = Path(self.tmp.name) / "existing.csv"
        target.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "已存在"):
            pipeline.run(str(EXAMPLES / "train.csv"), str(EXAMPLES / "predict.csv"), str(target))
        self.assertEqual(target.read_text(), "keep me")

    def test_empty_prediction_file_exports_headers(self):
        predict = Path(self.tmp.name) / "empty.csv"
        predict.write_text("用户原话\n")
        result = EncodingPipeline(self.config, FakeClient([])).run(
            str(EXAMPLES / "train.csv"), str(predict), str(Path(self.tmp.name) / "empty_result.csv")
        )
        self.assertEqual(len(result), 0)
        self.assertIn("需人工复核", result.columns)

    def test_formula_like_content_is_saved_as_text(self):
        from openpyxl import load_workbook
        path = Path(self.tmp.name) / "safe.xlsx"
        DataLoader.save_excel(pd.DataFrame({"用户原话": ["=1+1", "普通文本"]}), str(path))
        wb = load_workbook(path)
        self.assertEqual(wb.active["A2"].data_type, "s")
        self.assertEqual(wb.active["A2"].value, "'=1+1")
        wb.close()

    def test_no_hierarchy_mode_handles_valid_labels(self):
        self.learner.hierarchy_from_file = False
        client = FakeClient([{"一级编码": ["产品体验"], "二级编码": ["页面加载慢"], "confidence": "high"}])
        result = MultiLevelTextEncoder(self.config, self.learner, client).encode_single("页面慢")
        self.assertEqual(result["二级编码"], ["页面加载慢"])

    def test_failed_rows_are_not_counted_as_success(self):
        frame = pd.DataFrame({"预测_一级编码": ["编码失败", "产品体验"]})
        EncodingPipeline(self.config)._print_statistics(frame, ["一级编码"])
        self.assertIn("非空且非失败结果: 1 (50.0%)", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
