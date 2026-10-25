#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep docs/user_guide/examples/ in sync with examples/ on commit.

The pages under docs/user_guide/examples/ are generated from the README files
in examples/. mkdocs regenerates them at build time and CI fails when they are
stale, so contributors have to remember to run a docs build before pushing.

This hook does the same regeneration against the staged tree, without starting
mkdocs. It reuses docs/mkdocs/hooks/generate_examples.py, which is also what
mkdocs.yml loads.

Behaviour:

* docs/user_guide/examples/ staged without examples/: the edit is rejected,
  because those pages are machine-written and any manual change is lost on the
  next generation;
* docs/.nav.yml staged without examples/: rejected only when the edit reaches
  the User Guide -> Examples block, the one block the generator owns. A
  navigation entry for a hand-written page is kept by the generator and is
  allowed;
* something under examples/ staged: the pages are regenerated and staged, so the
  commit carries the matching docs;
* nothing relevant staged: exit quietly.

Set SKIP=sync-example-docs to disable it, for example when rebasing a branch
that is already green.
"""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

EXAMPLES_PREFIX = "examples/"
GENERATED_PAGE_PREFIX = "docs/user_guide/examples/"
NAV_FILE = "docs/.nav.yml"
GENERATED_PREFIXES = (GENERATED_PAGE_PREFIX, NAV_FILE)
GENERATOR_HOOK_DIR = "docs/mkdocs/hooks"
NAV_ROOT_SECTION = "User Guide"
NAV_GENERATED_SECTION = "Examples"


def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
    )
    return result.stdout


def repo_root() -> Path:
    """Return the working tree root."""
    return Path(git("rev-parse", "--show-toplevel").strip())


def nav_examples_section(text: str):
    """Return the ``User Guide -> Examples`` block of a nav document, or None.

    That block is the only part of docs/.nav.yml the generator rewrites
    (``generate_examples.update_nav_file``); every other section is preserved.
    """
    import yaml

    data = yaml.safe_load(text) or {}
    for item in data.get("nav") or []:
        if isinstance(item, dict) and NAV_ROOT_SECTION in item:
            for sub in item[NAV_ROOT_SECTION] or []:
                if isinstance(sub, dict) and NAV_GENERATED_SECTION in sub:
                    return sub[NAV_GENERATED_SECTION]
    return None


def nav_generated_block_changed(root: Path) -> bool:
    """True when the staged docs/.nav.yml edits the generated Examples block.

    Comparing the block rather than the whole file is what lets a contributor
    add a hand-written page and its nav entry: the generator preserves the
    sections it does not own, so an edit elsewhere is legitimate.
    """
    try:
        staged_text = git("show", f":{NAV_FILE}", cwd=root)
        head_text = git("show", f"HEAD:{NAV_FILE}", cwd=root)
    except subprocess.CalledProcessError:
        # New file, or no HEAD yet: nothing generated to compare against.
        return False
    return nav_examples_section(staged_text) != nav_examples_section(head_text)


def staged_files(root: Path) -> list[str]:
    """Return the paths staged for commit, relative to the repo root.

    Deletions are kept: a commit that only removes an example still has to
    regenerate the pages, and deleting a generated page is a violation like
    editing one. Both were missed while the filter dropped D.
    """
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMRD", cwd=root)
    return [line for line in out.splitlines() if line]


def diverging_generated_files(root: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (modified, new, deleted) generated files on disk vs the index."""
    pathspec = list(GENERATED_PREFIXES)
    modified = [line for line in git("diff", "--name-only", "--", *pathspec, cwd=root).splitlines() if line]
    new = [
        line
        for line in git("ls-files", "--others", "--exclude-standard", "--", *pathspec, cwd=root).splitlines()
        if line
    ]
    deleted = [line for line in git("ls-files", "--deleted", "--", *pathspec, cwd=root).splitlines() if line]
    return modified, new, deleted


def stale_generated_pages(root: Path) -> list[str]:
    """Return generated pages whose example is gone from examples/.

    ``on_startup`` writes the pages it finds but never removes the ones whose
    source disappeared, so deleting an example left its page behind pointing at
    nothing. The generator names a page ``<category>/<stem>.md`` after the
    example directory, so the source is missing when none of the shapes it
    reads (a directory, or a .md/.py/.sh file of that stem) exists.
    """
    stale: list[str] = []
    doc_root = root / GENERATED_PAGE_PREFIX
    if not doc_root.is_dir():
        return stale
    for page in sorted(doc_root.rglob("*.md")):
        relative = page.relative_to(doc_root)
        stem = root / EXAMPLES_PREFIX / relative.parent / relative.stem
        candidates = (
            stem,
            stem.with_suffix(".md"),
            stem.with_suffix(".py"),
            stem.with_suffix(".sh"),
        )
        if not any(candidate.exists() for candidate in candidates):
            stale.append(page.relative_to(root).as_posix())
    return stale


@contextmanager
def generator_importable(root: Path):
    """Make the mkdocs hook importable, then put sys.path back."""
    hook_dir = str(root / GENERATOR_HOOK_DIR)
    sys.path.insert(0, hook_dir)
    try:
        yield
    finally:
        sys.path.pop(0)


def regenerate(root: Path) -> None:
    """Run the same generation mkdocs runs at build time.

    The generator imports ``regex`` and ``yaml``. If either is missing, the
    hook's ``additional_dependencies`` are incomplete — that is a bug in this
    hook's setup, not in the contributor's commit, so say so plainly instead of
    letting a bare ModuleNotFoundError surface.
    """
    try:
        with generator_importable(root):
            import generate_examples

            generate_examples.on_startup("build", False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"the mkdocs generator needs '{exc.name}', which this hook does not "
            f"install. Add it to additional_dependencies under the "
            f"'sync-example-docs' entry in .pre-commit-config.yaml."
        ) from exc


def main() -> int:
    root = repo_root()
    staged = staged_files(root)

    examples_staged = [f for f in staged if f.startswith(EXAMPLES_PREFIX)]
    pages_staged = [f for f in staged if f.startswith(GENERATED_PAGE_PREFIX)]
    nav_staged = NAV_FILE in staged

    # The generated pages are always off-limits without an examples/ change.
    # docs/.nav.yml is only off-limits when the staged edit touches the block
    # the generator owns; a navigation entry for a hand-written page is fine.
    blocked = list(pages_staged)
    if nav_staged and not examples_staged and nav_generated_block_changed(root):
        blocked.append(NAV_FILE)

    if blocked and not examples_staged:
        print(
            "error: these files are generated, edit the sources in examples/ instead:",
            file=sys.stderr,
        )
        for path in blocked:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nThe next generation overwrites anything typed here. Change the "
            "README under examples/ and commit it; this hook updates the pages.",
            file=sys.stderr,
        )
        return 1

    if not examples_staged:
        return 0

    try:
        regenerate(root)
    except Exception as exc:  # noqa: BLE001 - surface any generator failure to the user
        print(
            f"error: generating the example pages failed: {exc}",
            file=sys.stderr,
        )
        return 1

    modified, new, deleted = diverging_generated_files(root)
    orphaned = stale_generated_pages(root)
    if not modified and not new and not deleted and not orphaned:
        return 0

    if modified or new:
        subprocess.run(
            ["git", "add", "--", *GENERATED_PREFIXES],
            cwd=root,
            check=True,
        )
    if deleted or orphaned:
        subprocess.run(["git", "rm", "-f", "--", *deleted, *orphaned], cwd=root, check=True)

    print("Synced the example pages with examples/:")
    for path in modified:
        print(f"  modified: {path}")
    for path in new:
        print(f"  new:      {path}")
    for path in deleted:
        print(f"  removed:  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
