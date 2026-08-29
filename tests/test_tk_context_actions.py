from tk_context_actions import context_action_labels


def test_ina_gui_context_actions_benchmark_v1_keyboard_only_vs_v2_right_click():
    """V2 exposes useful actions while respecting editable/read-only surfaces."""
    read_only = context_action_labels(has_selection=True, editable=False, has_text=True)
    editable = context_action_labels(has_selection=True, editable=True, has_text=True)

    assert read_only == ["Copy", "Copy all", "Select all"]
    assert editable == ["Copy", "Copy all", "Cut", "Paste", "Select all"]


def test_empty_read_only_surface_has_no_misleading_actions():
    assert context_action_labels(
        has_selection=False, editable=False, has_text=False,
    ) == []
