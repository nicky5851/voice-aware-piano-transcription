"""Ground-truth isolation: only src/vapt/eval/final.py may touch test-split MIDI (protocol §13.1)."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PATHS = {
    REPO_ROOT / "src" / "vapt" / "eval" / "final.py",
    REPO_ROOT / "src" / "vapt" / "mixing" / "manifests.py",  # reads test audio RMS only, never MIDI
    REPO_ROOT / "scripts" / "build_manifests.py",
}

TEST_MANIFEST_PATTERN = re.compile(r"test_manifest\.json|manifests/test_manifest")
TEST_SPLIT_PATTERN = re.compile(r"""split\s*==\s*["']test["']""")
MIDI_INDICATORS = ("midi_event", "pretty_midi", "note_events", "TargetProcessor", ".mid'", '.mid"')


def violations(path: Path) -> list[str]:
    text = path.read_text()
    issues = []
    if TEST_MANIFEST_PATTERN.search(text):
        issues.append("references the test manifest path")
    if TEST_SPLIT_PATTERN.search(text) and any(ind in text for ind in MIDI_INDICATORS):
        issues.append("combines split=='test' with MIDI loading")
    return issues


def _scanned_files():
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        if root.exists():
            yield from sorted(root.rglob("*.py"))


def test_isolation_static_check_passes():
    found = {}
    for path in _scanned_files():
        if path in ALLOWED_PATHS:
            continue
        issues = violations(path)
        if issues:
            found[str(path.relative_to(REPO_ROOT))] = issues
    assert not found, f"ground-truth isolation violated: {found}"


def test_isolation_catches_planted_violation(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text(
        "import pretty_midi\n"
        "def load(split):\n"
        "    if split == 'test':\n"
        "        return pretty_midi.PrettyMIDI('x.mid')\n"
    )
    assert violations(bad)


def test_isolation_catches_planted_test_manifest_reference(tmp_path):
    bad = tmp_path / "bad_module2.py"
    bad.write_text("PATH = 'manifests/test_manifest.json'\n")
    assert violations(bad)
