from retrieve import ask_pdf, hybrid_search, llm_client

# 1. The Golden Dataset (Ground Truth)
golden_dataset = [
    {
        "question": "What is the 'most constrained variable' heuristic?",
        "ground_truth": "It chooses the variable with the fewest legal values."
    },
    {
        "question": "What is the main idea behind forward checking?",
        "ground_truth": "It keeps track of remaining legal values for unassigned variables and terminates the search when any variable has no legal values."
    },
    {
        "question": "What does a binary CSP constraint graph look like?",
        "ground_truth": "The nodes are variables, and the arcs are constraints."
    }
]


def grade_pipeline(question, generated_answer, context_chunks, ground_truth):
    """Uses the LLM to grade the RAG pipeline's performance."""

    context_string = "\n\n".join(context_chunks)

    # We write a prompt that forces the LLM to act as a strict grader
    grading_prompt = f"""You are an impartial grading system for an AI pipeline.
    You will be given a Question, the Ground Truth answer, the AI's Generated Answer, and the Context it retrieved.

    Evaluate the AI on two metrics:
    1. CONTEXT PRECISION: Does the 'Retrieved Context' actually contain the facts needed to answer the question? (Yes/No)
    2. FAITHFULNESS: Does the 'Generated Answer' match the 'Ground Truth', and is it fully supported by the context without hallucinating? (Yes/No)

    DATA:
    Question: {question}
    Ground Truth: {ground_truth}
    Retrieved Context: {context_string}
    Generated Answer: {generated_answer}

    Output your grade in exactly this format:
    Context Precision: [Yes/No]
    Faithfulness: [Yes/No]
    """

    try:
        completion = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": grading_prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Grading Error: {e}"


if __name__ == "__main__":
    print("--- Starting Automated RAG Evaluation ---\n")

    for i, test in enumerate(golden_dataset):
        print(f"Test {i + 1}: {test['question']}")

        # 1. Run the pipeline
        retrieved_context = hybrid_search(test['question'], top_k=5)
        bot_answer = ask_pdf(test['question'])

        # 2. Grade the pipeline
        grade = grade_pipeline(
            question=test['question'],
            generated_answer=bot_answer,
            context_chunks=retrieved_context,
            ground_truth=test['ground_truth']
        )

        print(f"Bot Answer: {bot_answer}")
        print(f"--- Report Card ---\n{grade}\n")
        print("=" * 50 + "\n")