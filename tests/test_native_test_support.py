import ina_test
from module_benchmarks import benchmark_module


def test_native_approx_raises_and_monkeypatch(monkeypatch):
    assert 0.30000000001 == ina_test.approx(0.3)
    with ina_test.raises(ValueError, match="specific") as caught:
        raise ValueError("specific error")
    assert isinstance(caught.value, ValueError)
    monkeypatch.setenv("INA_NATIVE_TEST", "yes")


def test_native_test_benchmark_runs_parameterized_fixture_case():
    v1, v2 = benchmark_module("native_test_support")
    assert v2.accuracy > v1.accuracy
    assert v2.component_scores["runner"] == {"correct": 1, "total": 1}
