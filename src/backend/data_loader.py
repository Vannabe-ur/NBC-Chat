import json
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nbc_data.json"

@dataclass
class QARecord:
    id: int
    category: str
    question: str
    answer: str

    @property
    def embed_text(self) -> str:
        # Question-only embedding tends to match user queries better
        return self.question

    @property
    def full_text(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


def load_nbc_dataset(path: str = DATA_DIR) -> list[QARecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    qa_pairs = data["dataset"]["qa_pairs"]

    records = [
        QARecord(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            answer=item["answer"],
        )
        for item in qa_pairs
    ]
    return records


if __name__ == "__main__":
    records = load_nbc_dataset()
    print(f"Loaded {len(records)} QA records")
    print(records[0])