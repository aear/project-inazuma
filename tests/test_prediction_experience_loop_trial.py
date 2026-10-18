from benchmarks.benchmark_prediction_experience_loop import run_trial


def test_existing_cycle_and_hindsight_support_ordered_feedback(tmp_path):
    fixed=run_trial(tmp_path/'fixed','V1')
    revised=run_trial(tmp_path/'revised','V2')
    assert fixed['phase_mae']['held_out'] == .4
    assert revised['phase_mae']['held_out'] == 0.0
    assert revised['phase_mae']['shifted'] > fixed['phase_mae']['shifted']
    assert revised['cycle_count'] == 12
    assert revised['final_choice'] == 'stop'
    assert not revised['trained_on_held_out']
    assert not revised['live_ina_learning']
    import json
    records=[json.loads(line) for line in (tmp_path/'revised'/'fixture-domain-events.jsonl').read_text().splitlines()]
    for forecast,observation in zip(records[::2],records[1::2]):
        assert forecast['kind']=='forecast'
        assert observation['kind']=='observation'
        assert observation['forecast_reference'] == f"fixture-event:{forecast['sequence']}"
        assert observation['sequence'] > forecast['sequence']


def test_hindsight_uses_explicit_observation_and_keeps_next_prediction_separate():
    from benchmarks.benchmark_prediction_experience_loop import evaluator
    current=evaluator(current=True)
    predicted={'predicted_vector':{'clarity':.2}}
    # An ordinary next forecast is not a world observation.
    assert current.evaluate_claims(predicted,{'predicted_vector':{'clarity':.9}}) == {}
    assert current.evaluate_claims(predicted,{'observed_vector':{'clarity':.7}})['clarity']['actual'] == .7
    assert current.evaluate_claims({**predicted,'observed_vector':{'clarity':.6}},
                                   {'predicted_vector':{'clarity':.9}})['clarity']['actual'] == .6


def test_hindsight_saves_pending_updates_without_scoring_or_trust_changes():
    from benchmarks.benchmark_prediction_experience_loop import measure_hindsight_semantics
    assert all(measure_hindsight_semantics().values())


def test_feedback_boundary_is_versioned():
    from module_benchmarks import benchmark_module
    rows=benchmark_module('prediction_feedback')
    assert [row.version for row in rows] == ['V1','V2']
    assert rows[1].correct == rows[1].total
    assert rows[0].correct < rows[0].total


def test_pending_link_storage_is_bounded_and_does_not_rewrite_forecasts(tmp_path):
    import json
    from benchmarks.benchmark_prediction_experience_loop import evaluator
    judge=evaluator(current=True,lifecycle=True)
    judge.prediction_updates_path=tmp_path/'prediction_updates.json'
    original=tmp_path/'prediction_log.json'
    original.write_text('original predictor-owned data')
    records=[{'timestamp':str(i),'prediction_update':{'reference':str(i+1)}} for i in range(101)]
    judge.save_prediction_updates(records)
    saved=json.loads(judge.prediction_updates_path.read_text())
    assert len(saved['updates']) == 100
    assert not saved['contains_observed_outcomes']
    assert saved['updates'][0]['prediction_reference'] == '1'
    assert original.read_text() == 'original predictor-owned data'
