# Evaluating answers

Faithfulness measures whether every claim in an answer is supported by the retrieved passages. An answer that adds facts from the model's own memory scores low on faithfulness.

Answer relevance measures whether the answer actually addresses the question that was asked. A correct but off-topic answer scores low on answer relevance.

Evaluation runs against a frozen set of questions so that two runs can be compared. A release is blocked when the average score falls below the agreed threshold.
