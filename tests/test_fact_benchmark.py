from mathllm.pretraining.fact_benchmark import (
    fact_eval_cases,
    fact_eval_texts,
    fact_seen_template_cases,
    fact_training_texts,
    make_facts,
)


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


def test_seen_probe_uses_training_question_forms_without_answer_leakage():
    facts = make_facts(64, 23)
    cases = fact_seen_template_cases(64, 31, facts)
    assert all("vocation" in prompt for prompt, _ in cases)
    assert all(answer not in prompt for prompt, answer in cases)


def test_heldout_loss_corpus_uses_all_disjoint_templates_without_duplicates():
    facts = make_facts(4, 23)
    records = fact_eval_texts(100, 31, facts)
    assert len(records) == 12
    assert len(set(records)) == len(records)
    assert all("What vocation does" not in record for record in records)
