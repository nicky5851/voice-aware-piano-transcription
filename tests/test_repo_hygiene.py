"""Static check that no internal-only, agent-instruction-style file is tracked.

See CLAUDE.md (dev workspace, not this repository) SS2.4: this repository
never contains agent instruction files, specs, execution plans, walkthroughs,
worklogs or similarly internal-only development material.
"""
import re
import subprocess

FORBIDDEN_PATTERNS = [
    r"(^|/)claude(\.local)?\.md$",
    r"(^|/)agents\.md$",
    r"(^|/)gemini\.md$",
    r"(^|/)\.claude(/|$)",
    r"(^|/)\.cursor(/|$)",
    r"(^|/)\.aider",
    r"(^|/)specs(/|$)",
    r"(^|/)walkthroughs(/|$)",
    r"(^|/)worklog",
    r"(^|/)notes/agent",
    r"coding_specs?[^/]*\.md$",
    r"execution_plan[^/]*\.md$",
    r"protocol_v\d+\.md$",
]


def test_no_forbidden_paths_tracked():
    tracked = subprocess.check_output(["git", "ls-files"]).decode().splitlines()
    bad = [
        path
        for path in tracked
        for pattern in FORBIDDEN_PATTERNS
        if re.search(pattern, path, re.IGNORECASE)
    ]
    assert not bad, f"forbidden path(s) tracked: {bad}"
