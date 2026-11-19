"""Static check that no internal-only, agent-instruction-style file is tracked.

See CLAUDE.md (dev workspace, not this repository) SS2.4: this repository
never contains agent instruction files, specs, execution plans, walkthroughs,
worklogs or similarly internal-only development material.
"""
import re
import subprocess

import pytest

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
    # Official runs ship a git-free tree (`git archive`, CLAUDE.md SS2.1.3):
    # this check only applies to an actual git checkout, where the pre-commit
    # and pre-push hooks already enforce it before anything is ever shipped.
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files"], stderr=subprocess.DEVNULL
        ).decode().splitlines()
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout (e.g. a shipped `git archive` tree)")
    bad = [
        path
        for path in tracked
        for pattern in FORBIDDEN_PATTERNS
        if re.search(pattern, path, re.IGNORECASE)
    ]
    assert not bad, f"forbidden path(s) tracked: {bad}"
