from mathllm.pretraining.fact_benchmark import fact_eval_cases, fact_training_texts, make_facts


def test_fact_generator_is_deterministic_and_has_unique_bindings():
    first = make_facts(256, 17)
    assert first == make_facts(256, 17)
    assert len({fact.entity for fact in first}) == 256
    assert len({fact.value for fact in first}) == 256


def test_train_and_eval_use_disjoint_question_wording_without_local_copying():
    facts = make_facts(64, 23)
    train = fact_training_texts(500, 29, facts)
    cases = fact_eval_cases(64, 31, facts)
    assert any(text.startswith("Question: What vocation") for text in train)
    assert all("Identify" in prompt or "What work" in prompt or "works as what" in prompt for prompt, _ in cases)
    assert all(answer not in prompt for prompt, answer in cases)
