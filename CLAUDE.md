# Claude Code instructions

Before making changes, read these files in order:

1. `AGENTS.md` — mandatory project and safety constraints.
2. `docs/HANDOFF.md` — exact current state, blockers, and future work.
3. `README.md` — setup, layout, and verified results.
4. `CONTRIBUTING.md` — checks and pull-request expectations.

Treat `AGENTS.md` as authoritative if documentation conflicts. Do not access, print, copy, or commit credentials. You may continue b4 only in the ordered workflow in `docs/HANDOFF.md`: test, patch the infrastructure issue, compile, dry-run, inspect, then submit one controlled b4 job. Stop on any failure and do not retry automatically. After all four models complete, follow `docs/ROADMAP.md` and report only metrics read from real artifacts.
