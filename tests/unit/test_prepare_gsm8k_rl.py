import pytest

from tools.prepare_gsm8k_rl_from_sft import convert_messages


def test_convert_messages_matches_verl_gsm8k_contract():
    result = convert_messages(
        [
            {"role": "user", "content": "What is 6 * 7?"},
            {
                "role": "assistant",
                "content": "Six groups of seven are 42.\n#### 42",
            },
        ],
        split="train",
        index=3,
    )

    assert result == {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": "What is 6 * 7?"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "42"},
        "extra_info": {
            "split": "train",
            "index": 3,
            "answer": "Six groups of seven are 42.\n#### 42",
            "question": "What is 6 * 7?",
        },
    }


def test_convert_messages_rejects_rows_without_a_gsm8k_answer():
    with pytest.raises(ValueError, match="final answer"):
        convert_messages(
            [
                {"role": "user", "content": "What is 6 * 7?"},
                {"role": "assistant", "content": "The answer is 42."},
            ],
            split="test",
            index=0,
        )
