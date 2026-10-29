import json
from pathlib import Path

from app.schemas import EmailRequest
from app.services.llm_triage import triage_with_llm


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
OUTPUT_PATH = BASE_DIR / "eval_results.json"


def calculate_metrics(expected, predicted, positive_label):
    """Calculate precision and recall for one positive label."""

    true_positive = sum(
        e == positive_label and p == positive_label
        for e, p in zip(expected, predicted)
    )

    false_positive = sum(
        e != positive_label and p == positive_label
        for e, p in zip(expected, predicted)
    )

    false_negative = sum(
        e == positive_label and p != positive_label
        for e, p in zip(expected, predicted)
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    return precision, recall


def calculate_accuracy(expected, predicted):
    """Calculate classification accuracy."""

    if not expected:
        return 0.0

    correct = sum(
        e == p for e, p in zip(expected, predicted)
    )

    return correct / len(expected)


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    results = []

    for item in dataset:
        print(f"\nEvaluating email {item['id']}...")

        email = EmailRequest(
            sender=item["sender"],
            subject=item["subject"],
            body=item["body"],
        )

        # The callback captures latency, token usage and cost
        # without changing the API's response schema.
        call_metrics = {}

        try:
            result = triage_with_llm(
                email,
                metrics_callback=call_metrics.update,
            )

            category_ok = (
                result.category == item["expected_category"]
            )

            priority_ok = (
                result.priority == item["expected_priority"]
            )

            record = {
                "id": item["id"],
                "subject": item["subject"],
                "category_expected": item["expected_category"],
                "category_predicted": result.category,
                "category_ok": category_ok,
                "priority_expected": item["expected_priority"],
                "priority_predicted": result.priority,
                "priority_ok": priority_ok,
                "summary": result.summary,
                "status": "success",
                "metrics": call_metrics,
            }

            print(
                f"Category: {'PASS' if category_ok else 'FAIL'} | "
                f"Priority: {'PASS' if priority_ok else 'FAIL'}"
            )

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

            record = {
                "id": item["id"],
                "subject": item["subject"],
                "category_expected": item["expected_category"],
                "category_predicted": None,
                "category_ok": False,
                "priority_expected": item["expected_priority"],
                "priority_predicted": None,
                "priority_ok": False,
                "summary": None,
                "status": "error",
                "error_type": type(exc).__name__,
                "metrics": call_metrics,
            }

        results.append(record)

        # Persist after every example so partial progress is retained.
        OUTPUT_PATH.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    # Exclude failed calls from the prediction lists for clarity.
    # Failed calls are reported separately and count as failures
    # in the overall accuracy denominator below.
    successful_results = [
        r for r in results if r["status"] == "success"
    ]

    expected_categories = [
        r["category_expected"] for r in successful_results
    ]
    predicted_categories = [
        r["category_predicted"] for r in successful_results
    ]

    expected_priorities = [
        r["priority_expected"] for r in successful_results
    ]
    predicted_priorities = [
        r["priority_predicted"] for r in successful_results
    ]

    total = len(results)
    successful = len(successful_results)
    failed = total - successful

    category_correct = sum(
        r["category_ok"] for r in results
    )
    priority_correct = sum(
        r["priority_ok"] for r in results
    )

    category_accuracy = category_correct / total if total else 0.0
    priority_accuracy = priority_correct / total if total else 0.0

    work_precision, work_recall = calculate_metrics(
        expected_categories,
        predicted_categories,
        "work",
    )

    high_precision, high_recall = calculate_metrics(
        expected_priorities,
        predicted_priorities,
        "high",
    )

    # Aggregate telemetry from calls where usage was available.
    metrics_list = [
        r["metrics"]
        for r in results
        if r.get("metrics")
    ]

    latencies = [
        m["latency_ms"]
        for m in metrics_list
        if m.get("latency_ms") is not None
    ]

    input_tokens = sum(
        m.get("input_tokens", 0)
        for m in metrics_list
    )

    output_tokens = sum(
        m.get("output_tokens", 0)
        for m in metrics_list
    )

    total_tokens = input_tokens + output_tokens

    estimated_cost = sum(
        m.get("estimated_cost", 0.0)
        for m in metrics_list
    )

    print("\n" + "=" * 45)
    print("       INBOXPILOT EVALUATION SUMMARY")
    print("=" * 45)

    print(f"Total emails:          {total}")
    print(f"Successful calls:      {successful}")
    print(f"Failed calls:          {failed}")

    print("\n--- Classification Quality ---")
    print(f"Category accuracy:     {category_accuracy:.1%}")
    print(f"Priority accuracy:     {priority_accuracy:.1%}")

    print(f"Work precision:        {work_precision:.1%}")
    print(f"Work recall:           {work_recall:.1%}")

    print(f"High priority precision: {high_precision:.1%}")
    print(f"High priority recall:    {high_recall:.1%}")

    print("\n--- Performance & Usage ---")

    if latencies:
        average_latency = sum(latencies) / len(latencies)

        print(f"Average latency:       {average_latency:.2f} ms")
        print(f"Minimum latency:       {min(latencies):.2f} ms")
        print(f"Maximum latency:       {max(latencies):.2f} ms")
    else:
        print("Latency:               No measurements available")

    print(f"Input tokens:          {input_tokens}")
    print(f"Output tokens:         {output_tokens}")
    print(f"Total tokens:          {total_tokens}")
    print(f"Estimated total cost:  {estimated_cost:.8f}")

    print("\nDetailed results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()