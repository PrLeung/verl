from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import re
from qwen_vl_utils import process_vision_info
import json
import sys
from typing import List, Dict, Any

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/vlm/peirouliang/checkpoints/qwen25_vl_7b_rl_cot_40k_vqa_20k_stage1", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/vlm/peirouliang/checkpoints/qwen25_vl_7b_rl_cot_40k_vqa_20k_stage1")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def _extract_question_and_prompt(sample: Dict[str, Any]) -> str:
    # 兼容两种形式：
    # 1) prompt 为字符串（如 sample_200.json）
    # 2) prompt 为消息列表（原有格式）
    prompt_field = sample.get("prompt", [])
    merged: str = ""
    if isinstance(prompt_field, str):
        merged = prompt_field
    else:
        prompt_items: List[Dict[str, Any]] = prompt_field or []
        texts: List[str] = []
        for item in prompt_items:
            text_piece = item.get("content", "")
            if isinstance(text_piece, str):
                texts.append(text_piece)
        merged = " ".join(texts).strip()

    # 去掉占位符
    merged = (merged or "").replace("<image>", "").strip()
    # 如果有更标准的问题字段，则优先
    question = sample.get("extra_info", {}).get("question") or merged
    return question or ""


def _extract_answer(sample: Dict[str, Any]) -> str:
    # 优先使用顶层 answer（如 sample_200.json），兼容原有字段
    top = sample.get("answer")
    if isinstance(top, str) and top:
        return top
    extra = sample.get("extra_info", {})
    rm = sample.get("reward_model", {})
    return extra.get("answer") or rm.get("ground_truth") or ""
def _extract_answer_tag_content(predict_str: str) -> str | None:
    pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(predict_str)
    return match.group(1) if match else None



def _extract_image_path(sample: Dict[str, Any]) -> str:
    images: List[str] = sample.get("images", [])
    if not images:
        return ""
    path = images[0]
    # 模型消息需要 file:// 前缀
    if not path.startswith("file://"):
        path = f"file://{path}"
    return path


def run_inference_on_dataset(json_path: str, max_new_tokens: int = 4096, output_path: str = "") -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层应为列表，每个元素是一个样本对象")

    results: List[Dict[str, Any]] = []
    num_correct = 0
    num_total = 0
    # 按数据源累计统计
    source_stats: Dict[str, Dict[str, int]] = {}
    for idx, sample in enumerate(data, start=1):
        image_uri = _extract_image_path(sample)
        question = _extract_question_and_prompt(sample)
        answer = _extract_answer(sample)
        data_source = sample.get("data_source", "")

        if not image_uri or not question:
            print(f"[跳过] 第{idx}条样本，缺少图片或问题")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # 准备输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 推理
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        pred = output_text[0] if output_text else ""

        # 评测：提取 <answer>...</answer> 的内容并与参考答案对比（lower+strip）
        extracted_pred = _extract_answer_tag_content(pred) or ""
        pred_norm = extracted_pred.lower().strip()
        ref_norm = (answer or "").lower().strip()
        is_correct = pred_norm == ref_norm if ref_norm else False
        num_total += 1
        if is_correct:
            num_correct += 1

        # 记录按数据源的统计
        source_key = data_source or ""
        if source_key not in source_stats:
            source_stats[source_key] = {"correct": 0, "total": 0, "think": 0, "no_think": 0}
        source_stats[source_key]["total"] += 1
        if is_correct:
            source_stats[source_key]["correct"] += 1

        # 统计生成内容中是否包含 <think> 或 <no_think>
        pred_lc = (pred or "").lower()
        if "<think>" in pred_lc:
            source_stats[source_key]["think"] += 1
        elif "<no_think>" in pred_lc:
            source_stats[source_key]["no_think"] += 1

        print("==== 样本", idx, "====")
        print("数据源:", data_source)
        print("问题:", question)
        print("参考答案:", answer)
        print("模型输出:", pred)
        print("提取答案:", extracted_pred)
        print("是否正确:", is_correct)
        print()

        results.append(
            {
                "index": idx,
                "data_source": data_source,
                "question": question,
                "answer": answer,
                "prediction": pred,
                "pred_extracted": extracted_pred,
                "correct": is_correct,
                "image": sample.get("images", [""])[0] if sample.get("images") else "",
                "sample_id": sample.get("extra_info", {}).get("id", ""),
            }
        )

    # 保存为 JSON 文件（文件名包含模型名称）
    if not output_path:
        # 获取模型名称并做文件系统安全处理
        model_name = getattr(model, "name_or_path", None) or getattr(getattr(model, "config", object()), "_name_or_path", "model")
        model_name = str(model_name)
        # 替换路径分隔符与空白为下划线，去除无效字符
        safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        # 基于输入文件名生成输出前缀
        if json_path.endswith(".json"):
            base = json_path[:-5]
        elif json_path.endswith(".jsonl"):
            base = json_path[:-6]
        else:
            base = json_path
        output_path = f"{base}.{safe_model_name}.pred.json"
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    print(f"已保存结果到: {output_path}")

    # 打印整体准确率
    if num_total > 0:
        acc = num_correct / num_total
        print(f"数据集准确率: {acc:.4f}  ({num_correct}/{num_total})")

    # 打印各数据源准确率
    if source_stats:
        print("\n按数据源统计:")
        for src, cnts in sorted(source_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            c = cnts["correct"]
            t = cnts["total"]
            acc = (c / t) if t else 0.0
            th = cnts.get("think", 0)
            nth = cnts.get("no_think", 0)
            think_ratio = (th / t) if t else 0.0
            no_think_ratio = (nth / t) if t else 0.0
            src_name = src if src else "<EMPTY>"
            print(f"- {src_name}: 准确率 {acc:.4f}  ({c}/{t}); THINK占比 {think_ratio:.4f}  ({th}/{t}); NO_THINK占比 {no_think_ratio:.4f}  ({nth}/{t})")


if __name__ == "__main__":
    default_json = "/vlm/peirouliang/verl/data/sample_200.json"
    json_path = sys.argv[1] if len(sys.argv) > 1 else default_json
    output_path = sys.argv[2] if len(sys.argv) > 2 else ""
    run_inference_on_dataset(json_path, output_path=output_path)
