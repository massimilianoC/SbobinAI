# Security policy

SbobinAI processes local files and talks to a local inference server
bound to the loopback interface. Please report vulnerabilities privately.

## Reporting

Use GitHub's **private vulnerability reporting** (Security → Report a
vulnerability) on this repository. If that is not possible, write to
**massimiliano@camillucci.com**. Do not open a public issue.
Include the affected version, a description, reproduction steps and the impact.
Please do not include private recordings or transcripts.

## Scope notes

- The llama.cpp adapter only accepts loopback server URLs by design.
- Downloads happen only through explicit setup commands and are verified by SHA-256.
- Media, transcripts and logs stay in local, git-ignored folders.
