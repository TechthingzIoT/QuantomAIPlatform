# Security Policy

Thank you for helping keep **QuantomAIPlatform** and the **Quantom AI Runtime (QAIR)** secure.

We take the security of our software seriously and appreciate responsible disclosure of vulnerabilities.

---

# Supported Versions

The following versions currently receive security updates:

| Version | Supported |
|----------|-----------|
| v0.1.x | ✅ Yes |
| Older versions | ❌ No |

As the project matures, this table will be updated for each major release.

---

# Reporting a Vulnerability

If you discover a security vulnerability, **please do not create a public GitHub issue.**

Instead, report it privately by contacting the project maintainers.

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested remediation (if known)
- Screenshots or logs (if applicable)

Providing as much information as possible helps us investigate and resolve issues more quickly.

---

# Response Process

Once a report is received, we aim to:

### Within 72 hours

- Acknowledge receipt of the report.

### Within 7 days

- Assess the issue.
- Determine severity.
- Begin remediation if confirmed.

### Before Public Disclosure

- Develop and test a fix.
- Notify affected users when appropriate.
- Publish a security advisory after the fix is available.

---

# Security Scope

Security reports may include, but are not limited to:

- Authentication flaws
- Authorization bypasses
- Remote code execution
- Command injection
- Path traversal
- Arbitrary file access
- Sensitive data exposure
- Dependency vulnerabilities
- API security issues
- Prompt injection affecting protected functionality
- Plugin sandbox escapes
- Model execution vulnerabilities

---

# Out of Scope

The following are generally not considered security vulnerabilities:

- Minor spelling or documentation errors
- UI or formatting issues
- Theoretical attacks without practical impact
- Vulnerabilities already publicly disclosed and under active remediation

---

# Dependency Security

QAIR relies on third-party software such as:

- Python packages
- llama.cpp
- AI model runtimes
- Operating system libraries

We regularly review and update dependencies to address known vulnerabilities.

---

# Secrets Management

Never commit sensitive information to the repository, including:

- API keys
- Passwords
- SSH private keys
- Tokens
- Certificates
- Cloud credentials
- Database passwords

Use environment variables or secure secret-management solutions instead.

Example:

```bash
OPENAI_API_KEY=xxxxxxxxxxxxxxxx
```

Never hard-code secrets in source code.

---

# Responsible Disclosure

We ask that researchers:

- Give us reasonable time to investigate.
- Avoid publicly disclosing vulnerabilities before a fix is available.
- Refrain from accessing, modifying, or deleting user data.
- Avoid disrupting production systems.

We are committed to working collaboratively with the security community.

---

# Security Best Practices

Contributors should:

- Keep dependencies updated.
- Run security scans before releases.
- Validate all user input.
- Follow secure coding practices.
- Write tests for security-sensitive features.

---

# Future Security Enhancements

As QAIR evolves, planned improvements include:

- Automated dependency scanning
- Static application security testing (SAST)
- Secret scanning
- Container image scanning
- Signed releases
- SBOM (Software Bill of Materials)
- Security advisories via GitHub

---

# Contact

Security-related inquiries should be directed to the project maintainers through the project's private communication channels.

As the project grows, a dedicated security contact address will be published.

---

# Acknowledgements

We appreciate the efforts of security researchers and contributors who help improve the safety and reliability of QuantomAIPlatform and QAIR through responsible disclosure.

---

**Document Version:** 1.0