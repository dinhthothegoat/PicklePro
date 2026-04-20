# Development Principles

These notes preserve the useful parts of the removed code-reviewer and prompt-instructor prototypes without exposing them as application features.

## Code Review

- Prioritize bugs, privacy leaks, security issues, data ownership mistakes, and missing tests before style.
- Check every user-facing data view for authorization and per-user scoping.
- Treat background jobs, uploads, external downloads, and model inference as failure-prone paths.
- Prefer focused regression tests for each fixed risk.
- Keep unrelated refactors out of behavior fixes.

## Prompting

- State the task, audience, constraints, and desired output format.
- Include enough context to avoid guessing, but keep instructions short.
- Ask for verification criteria when the result affects privacy, money, or user trust.
- Require the answer to call out uncertainty instead of pretending weak evidence is strong.
- Keep developer-only aids out of product navigation unless they are real user-facing features.
