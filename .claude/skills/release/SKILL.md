---
name: release
description: >
  Cut a new ferrum release: bump version across repo, generate changelog from conventional
  commits, update docs/site/changelog.md, create a GitHub release with tag. Triggered
  explicitly only — never auto-invoked. Use when the user says /release, "cut a release",
  "publish a new version", "bump version and release", or "release X.Y.Z".
---

# Release

Cut a versioned release of ferrum. User-triggered only.

## Usage

```
/release <version | patch | minor | major>
```

Examples: `/release 0.9.0`, `/release minor`, `/release patch`

---

## Steps

### 1. Parse version argument

- If arg is `patch`, `minor`, or `major`: read current version from `pyproject.toml`, compute next per semver.
- If arg is an explicit version string (e.g., `0.9.0`): validate it is valid semver and greater than current.
- If no arg provided: ask the user what version to release.

### 2. Version consistency — bump all locations

Update the version string in **both** of these files:

| File | Field |
|---|---|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `Cargo.toml` (workspace root) | `version = "X.Y.Z"` |

After bumping, grep the repo for the **old** version string to catch any other references (README badges, docs, etc.). If found, update them or warn the user.

### 3. Generate changelog from commits

Collect commits since the last git tag (or all commits if no prior tag exists). Group by conventional commit prefix:

| Prefix | Section |
|---|---|
| `feat` | **Added** |
| `fix` | **Fixed** |
| `refactor`, `perf` | **Changed** |
| `docs`, `ci`, `chore`, `test`, `build` | **Other** |
| No prefix / unknown | **Other** |

Rules:
- Skip merge commits.
- Strip the prefix and scope from the message for display (e.g., `feat(interactive): packed tooltips` → `packed tooltips`).
- One bullet per commit. Keep the one-liner — don't expand.
- Drop the **Other** section if empty; always include **Added** and **Fixed** even if empty (just say "None").

### 4. Update docs site changelog

Prepend a new release section to `docs/site/changelog.md`:
- Replace the `## Unreleased` content with `*No unreleased changes.*`
- Insert `## X.Y.Z` section (with today's date) between `## Unreleased` and the previous release, containing the generated changelog.

### 5. Present for confirmation

Show the user:
- New version number
- Files modified (with diffs if small)
- Full changelog draft

Ask: **"Ready to commit, tag, push, and create the GitHub release?"**

Do NOT proceed until the user confirms.

### 6. On approval — commit, tag, push, release

```bash
git add pyproject.toml Cargo.toml docs/site/changelog.md
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z" --notes "<changelog body>"
```

### 7. Done

Report the release URL. Remind the user that the `publish.yaml` workflow will build wheels and publish to PyPI automatically.

---

## Guardrails

- **Never auto-trigger.** This skill only runs when the user explicitly invokes it.
- **Never push without confirmation.** Step 5 is a hard gate.
- **Fail early on dirty working tree.** If there are uncommitted changes, abort and tell the user to commit or stash first.
- **Fail if on a branch other than `main`.** Releases come from main only (unless user overrides).
- **Fail if tests don't pass.** Run `uv run pytest` before proceeding. If tests fail, abort.
