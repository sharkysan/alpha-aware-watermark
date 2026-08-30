# Security Policy

## Supported versions

Security fixes are applied to the latest version on the `main` branch.

## Reporting a vulnerability

Please do not publish exploitable vulnerability details in a public issue.
Use GitHub's private vulnerability reporting feature for this repository when
available. If private reporting is unavailable, contact the repository owner
privately and provide:

- the affected version or commit;
- a concise description of the vulnerability;
- reproduction steps or a minimal proof of concept;
- the expected impact;
- any suggested mitigation, if known.

Reports will be assessed for reproducibility, severity, and affected scope.

## Automated security checks

The repository uses:

- CodeQL for source-code security analysis;
- Bandit for Python-specific static security checks;
- `pip-audit` for known vulnerabilities in Python dependencies;
- Dependabot for Python and GitHub Actions dependency updates.

## Third-party software

This project integrates with external tools such as ProPainter and may be used
with SAM 2 / SAM 2.1. Those projects are not covered by this repository's MIT
licence. Review and comply with each third-party project's licence and usage
terms before use or redistribution.
