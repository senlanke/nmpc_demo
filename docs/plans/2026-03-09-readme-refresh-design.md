# README Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the public repository README so new visitors can quickly understand, install, run, and test the standalone `nmpc_demo` project.

**Architecture:** Keep the change documentation-only. Reorganize the README around repository purpose, setup flow, run modes, project layout, and known limitations. Do not change runtime behavior or add speculative claims not supported by the repository.

**Tech Stack:** Markdown, Git, Python project structure, MuJoCo demo usage

---

### Task 1: Rewrite public-facing README structure

**Files:**
- Modify: `README.md`

**Step 1: Draft the new README structure**

Include these sections:
- `Overview`
- `Features`
- `Repository Layout`
- `Requirements`
- `Installation`
- `Running the Demo`
- `Headless Mode`
- `Tests`
- `Known Limitations`

**Step 2: Update content with repository-specific details**

Use only details visible in the project:
- standalone pure-Python xArm6 NMPC obstacle avoidance demo
- fixed `start_q`
- draggable MuJoCo mocap obstacle
- repository directories and entrypoint files

**Step 3: Verify Markdown readability**

Run: `sed -n '1,260p' README.md`
Expected: clear section order, commands render correctly, no placeholder text

**Step 4: Review diff**

Run: `git diff -- README.md`
Expected: documentation-only diff with improved structure and no unrelated file changes

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-03-09-readme-refresh-design.md
git commit -m "docs: refresh repository README"
```
