from typing import Tuple

DEFAULT_WINDOW_DURATION_US = 100_000   # 100 ms, used before a recording's first target frame


def normalize_event_view(event_view: str) -> Tuple[str, ...]:
    """Return the camera sides to load for `event_view` ('left', 'right' or 'stereo')."""
    normalized_view = event_view.lower()
    if normalized_view == "stereo":
        return ("left", "right")
    if normalized_view in {"left", "mono", "mono_left"}:
        return ("left",)
    if normalized_view in {"right", "mono_right"}:
        return ("right",)
    raise ValueError(
        f"event_view must be one of 'left', 'right', 'stereo', got {event_view!r}"
    )
