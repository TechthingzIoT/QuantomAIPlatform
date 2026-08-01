# Contributing to QuantomAIPlatform

First, thank you for your interest in contributing to **QuantomAIPlatform** and the **Quantom AI Runtime (QAIR)**.

Our goal is to build a world-class AI runtime for Robotics, IoT, Edge AI, Intelligent Automation, and AI Research. Every contribution, whether code, documentation, testing, or design, helps move the project forward.

---

# Our Philosophy

We value:

- High-quality engineering
- Clean, maintainable code
- Well-documented features
- Automated testing
- Respectful collaboration
- Continuous learning

Every contribution should improve the project without compromising stability or readability.

---

# Before You Start

Before contributing, please:

- Read the README.md
- Review the ARCHITECTURE.md
- Check the ROADMAP.md
- Search existing Issues before opening a new one

---

# Ways to Contribute

You can contribute by:

- Fixing bugs
- Implementing features
- Improving documentation
- Writing tests
- Improving performance
- Reviewing Pull Requests
- Reporting issues
- Suggesting enhancements

---

# Development Environment

Clone the repository:

```bash
git clone git@github.com:TechthingzIoT/QuantomAIPlatform.git

cd QuantomAIPlatform
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

---

# Branch Strategy

Never work directly on the **main** branch.

Create a feature branch:

```bash
git checkout -b feature/my-feature
```

Examples:

```text
feature/rag
feature/api
feature/robotics
feature/vision

bugfix/session

docs/readme

refactor/backend
```

---

# Commit Messages

Follow Conventional Commits.

Examples:

```text
feat: add backend abstraction

fix: resolve session memory leak

docs: update architecture guide

refactor: simplify prompt manager

test: improve inference coverage

chore: update dependencies
```

---

# Coding Standards

Python version:

```
Python 3.12+
```

Requirements:

- Use type hints
- Write descriptive variable names
- Keep functions focused
- Avoid unnecessary complexity
- Prefer composition over inheritance

---

# Formatting

Format code before committing:

```bash
black .
```

Lint:

```bash
ruff check .
```

Run tests:

```bash
pytest
```

All checks must pass before submitting a Pull Request.

---

# Testing

Every new feature should include tests.

Types of tests:

- Unit Tests
- Integration Tests
- Regression Tests

Test files belong in:

```text
tests/
```

---

# Documentation

If your change affects users, update:

- README.md
- ARCHITECTURE.md
- ROADMAP.md
- CHANGELOG.md

Documentation is considered part of the feature.

---

# Pull Requests

A Pull Request should:

- Focus on a single topic
- Include tests
- Update documentation where applicable
- Pass all CI checks

Before requesting review, verify:

- Code builds successfully
- Tests pass
- Documentation is updated

---

# Code Review

Reviewers will evaluate:

- Correctness
- Readability
- Maintainability
- Performance
- Security
- Test coverage
- Documentation

Feedback is expected to be constructive and respectful.

---

# Reporting Bugs

When reporting a bug, include:

- Operating System
- Python version
- QAIR version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages or logs

Screenshots are welcome when relevant.

---

# Feature Requests

A feature proposal should explain:

- The problem being solved
- The proposed solution
- Alternative approaches considered
- Expected benefits
- Potential drawbacks

---

# Security

Please do **not** report security vulnerabilities in public issues.

Refer to **SECURITY.md** for responsible disclosure procedures.

---

# Community Standards

We are committed to maintaining a welcoming, respectful, and inclusive community.

Please follow the project's **CODE_OF_CONDUCT.md** in all interactions.

---

# Questions

If you're unsure where to start:

- Check the ROADMAP.md
- Browse open issues
- Review project documentation
- Start with documentation or tests if you're new to the codebase

---

# Thank You

Thank you for helping build QuantomAIPlatform and QAIR.

Every contribution—large or small—helps strengthen the platform and its mission to advance AI, robotics, and intelligent automation.

---

**Document Version:** 1.0