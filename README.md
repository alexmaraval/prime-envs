# Prime Environments

This repository is a high-level workspace for developing multiple Prime/Verifiers environments. Each environment project lives in its own subdirectory and can keep its own package metadata, configs, tests, and documentation.

The root of this repo is intentionally lightweight. It exists to group related environment workspaces in one place rather than act as a single installable package.

## Repository Layout

Typical structure:

```text
prime-envs/
├── README.md
├── games/
│   └── example-environment/
│       ├── README.md
│       ├── pyproject.toml
│       ├── environments/
│       │   └── example_env/
│       ├── configs/
│       └── tests/
└── ...
```

Current example workspace:

- `games/hangman/`: local Prime environment workspace for a Hangman agent

## Conventions

Each environment subdirectory should be self-contained and usually include:

- a local `README.md` with task details and quickstart commands
- a `pyproject.toml` for package and dependency management
- an `environments/<env_id>/` package exposing the environment implementation
- tests and local config files as needed

Recommended approach:

- treat each subdirectory as its own environment workspace
- run install, eval, and test commands from the relevant subdirectory
- keep environment-specific dependencies local to that workspace
- document any custom setup directly in the subdirectory README

## Getting Started

1. Create a new subdirectory for the environment family or project you want to work on.
2. Add the package structure and environment code inside that workspace.
3. Include a `README.md` with quickstart, eval, and local development notes.
4. Validate the environment locally before treating it as reusable.

Example:

```text
games/
  my-new-environment/
```

Inside that workspace you would typically add:

```text
my-new-environment/
├── README.md
├── pyproject.toml
├── environments/
│   └── my_env/
└── tests/
```

## Working Model

This repo is best treated as a collection of environment workspaces, not a single monorepo with one shared runtime. That keeps each environment easy to iterate on, test, and publish independently.

If you want a starting point, use `games/hangman/` as a reference for how an individual environment workspace can be organized.
