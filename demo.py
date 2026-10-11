"""Offline walkthrough using fixed synthetic model responses, not real inference."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from user_text_encoder_v3 import EncoderConfig, EncodingPipeline


RESPONSES = [
    {"二级编码": ["页面加载慢", "客服响应慢"], "confidence": "high"},
    {"二级编码": ["优惠不足"], "confidence": "high"},
    {"二级编码": ["服务态度好", "功能难找"], "confidence": "medium"},
    {"二级编码": ["收费说明不清"], "confidence": "high"},
    {"二级编码": [], "confidence": "low"},
]


class DemoClient:
    """Return fixture responses in the same order as examples/predict.csv."""

    def __init__(self):
        self.responses = iter(RESPONSES)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(next(self.responses), ensure_ascii=False)
        ))])


def main():
    parser = argparse.ArgumentParser(description="离线演示：固定模拟响应，不调用模型")
    parser.add_argument("--output", default="outputs/demo_result.csv")
    args = parser.parse_args()
    examples = Path(__file__).resolve().parent / "examples"
    config = EncoderConfig(
        model="offline-fixture-not-a-model",
        hierarchy_file=str(examples / "hierarchy.csv"),
        question_context="你对这款示例产品的使用体验有什么意见或建议？",
        batch_size=100,
    )
    print("离线演示：所有数据均为虚构，模型响应为预设内容，不代表 AI 准确率。")
    EncodingPipeline(config, client=DemoClient()).run(
        str(examples / "train.csv"), str(examples / "predict.csv"), args.output
    )


if __name__ == "__main__":
    main()
