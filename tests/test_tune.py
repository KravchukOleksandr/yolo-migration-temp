class FakeTrial:
    def suggest_float(
        self, name: str, low: float, high: float, log: bool, step: float | None
    ) -> float:
        assert low < high
        assert not (log and step is not None)
        return low

    def suggest_int(
        self, name: str, low: int, high: int, step: int, log: bool
    ) -> int:
        assert low < high
        return low

    def suggest_categorical(self, name: str, choices: list) -> object:
        return choices[0]


def test_declarative_search_space() -> None:
    from yolo_factory.tune import suggest_parameters

    parameters = suggest_parameters(
        FakeTrial(),
        {
            "lr0": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
            "warmup": {"type": "int", "low": 1, "high": 3},
            "optimizer": {"type": "categorical", "choices": ["MuSGD"]},
        },
    )

    assert parameters == {"lr0": 0.001, "warmup": 1, "optimizer": "MuSGD"}
