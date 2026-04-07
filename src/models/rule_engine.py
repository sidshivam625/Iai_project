import re

class KnowledgeRuleEngine:
    def __init__(self):
        # r: (regex_pattern, weight)
        self.rules = [
            (r'(?i)ignore\s+previous\s+instructions', 0.8),
            (r'(?i)you\s+are\s+now\s+in\s+developer\s+mode', 0.9),
            (r'(?i)system\s+prompt', 0.6),
            (r'(?i)act\s+as\s+a\s+linux\s+terminal', 0.5),
            (r'(?i)bypass\s+filters', 0.8)
        ]

    def evaluate(self, prompt: str) -> float:
        score = 0.0
        for pattern, weight in self.rules:
            if re.search(pattern, prompt):
                score += weight
        # Cap rule score at 1.0 ideally, or let it accumulate depending on scaling
        return min(score, 1.0)
