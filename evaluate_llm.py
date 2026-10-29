import json
from pathlib import Path

from app.schemas import EmailRequest
from app.services.llm_triage import triage_with_llm

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

def calculate_metrics(expected, predicted, positive_label):
    tp = sum(
        e == positive_label and p == positive_label
        for e,p in zip(expected,predicted)
    )
    
    fp = sum(
        e!=positive_label and p==positive_label
        for e,p in zip(expected, predicted)
    )
    
    fn = sum(
        e==positive_label and p!=positive_label
        for e,p in zip(expected, predicted)
    )
    
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    
    return precision, recall

def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    
    results = []
    
    for item in dataset:
        email = EmailRequest(
            sender = item["sender"],
            subject = item["subject"],
            body = item["body"]
        )
        
        try:
            result = triage_with_llm(email)
            
            category_ok = (
                result.category == item["expected_category"]
            )
            priority_ok = (
                result.priority == item["expected_priority"]
            )
            
            results.append({
                "id": item["id"],
                "category_expected": item["expected_category"],
                "category_predicted": result.category,
                "category_ok": category_ok,
                "priority_expected": item["expected_priority"],
                "priority_predicted": result.priority,
                "priority_ok": priority_ok,
            })
            
            print(
                f"Email {item['id']}: "
                f"category={'PASS' if category_ok else 'FAIL'}, "
                f"priority={'PASS' if priority_ok else 'FAIL'}"
            )
        except Exception as exc:
            print(f"Email {item['id']}: ERROR ({type(exc).__name__})")
            results.append({
                "id": item["id"],
                "category_expected": item["expected_category"],
                "category_predicted": None,
                "category_ok": False,
                "priority_expected": item["expected_priority"],
                "priority_predicted": None,
                "priority_ok": False,
            })
               
    expected_categories = [r["category_expected"] for r in results]
    predicted_categories = [r["category_predicted"] for r in results]
    expected_priorities = [r["priority_expected"] for r in results] 
    predicted_priorities = [r["priority_predicted"] for r in results]
    
    def accuracy(expected, predicted):
        return sum(
          e == p for e,p in zip(expected, predicted)  
        ) / len(expected) if expected else 0.0 
        
    print("\n=== Evaluation Summary ===")
    print(f"Category accuracy: {accuracy(expected_categories, predicted_categories):.1%}")
    print(f"Priority accuracy: {accuracy(expected_priorities, predicted_priorities):.1%}")
    
    for label, expected, predicted in [
        ("Work category", expected_categories, predicted_categories),
        ("High priority", expected_priorities, predicted_priorities),
    ]:
        positive_label = "work" if label == "Work category" else "high"
        precision, recall = calculate_metrics(
            expected, predicted, positive_label
        )
        print(f"{label} precision: {precision:.1%}")
        print(f"{label} recall:    {recall:.1%}")

    output_path = Path(__file__).parent / "eval_results.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()           