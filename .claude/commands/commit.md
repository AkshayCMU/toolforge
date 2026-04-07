Run a safe conventional commit for this toolforge session.

Steps:
1. Run `git status` to see staged/unstaged changes.
2. Run `git diff --staged` to review exactly what will be committed.
3. Check that no secrets are staged (the pre-commit hook will enforce this, but scan for `sk-ant-`, `ANTHROPIC_API_KEY=`, `.env` being staged).
4. If nothing is staged yet, ask the user which files to stage.
5. Draft a conventional commit message following the project convention:
   - `feat(module): description` for new functionality
   - `fix(module): description` for bug fixes
   - `refactor:`, `test:`, `docs:`, `chore:` as appropriate
   - Keep the subject line under 72 characters
   - Each completed FEATURES.md feature = one commit
6. Create the commit with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` trailer.
7. Run `git log --oneline -3` to confirm the commit landed.
8. Report the commit hash and message to the user.

Never commit: `.env`, `runs/`, `artifacts/`, `reports/`, `.cache/`, `*.jsonl`, `*.pkl`.
Never use `--no-verify` — fix the issue instead.
