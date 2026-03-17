# Prime Environments

This repository is a high-level workspace for building and iterating on Prime/Verifiers environments. The main pattern is to keep each environment in its own subdirectory, with its own code, metadata, configs, and documentation.

The root of the repo stays intentionally lightweight. It is a container for environment workspaces, not a single installable package.

## Disclaimer

All environments in this repository are vibe-coded in collaboration with Codex.

## Repository Layout

The current structure is centered around root-level environment directories:

```text
prime-envs/
├── README.md
├── environments/
│   ├── gpt-world/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── configs/
│   │   └── ...
│   └── hangman/
│       ├── README.md
│       ├── pyproject.toml
│       └── ...
├── games/
│   └── hangman/
│       ├── README.md
│       ├── pyproject.toml
│       ├── environments/
│       └── tests/
└── ...
```

Current environment workspaces at the root:

- `environments/gpt-world/`: hex-grid pathfinding environment workspace
- `environments/hangman/`: hangman environment workspace

There is also an older project-style workspace at `games/hangman/`, but the root README should be read primarily through the `environments/` layout above.

## Conventions

Each environment directory should be self-contained and usually include:

- a local `README.md` with task details and quickstart commands
- a `pyproject.toml` with package metadata and dependencies
- the environment implementation files for that package
- optional `configs/`, tests, helper scripts, and local assets

Recommended approach:

- treat each environment directory as its own workspace
- run install, eval, and test commands from the relevant environment directory when needed
- keep environment-specific dependencies and notes local to that directory
- document environment arguments, metrics, and local workflows in the local README

## Getting Started

To add a new environment, create a new subdirectory under `environments/`:

```text
environments/
  my-new-environment/
```

A typical minimal workspace looks like:

```text
my-new-environment/
├── README.md
├── pyproject.toml
├── my_new_environment.py
└── configs/
```

Then:

1. Implement the environment package and expose the expected entrypoint.
2. Add a local README with quickstart and evaluation examples.
3. Keep any environment-specific configs or helper scripts alongside the environment.
4. Validate the environment locally before treating it as reusable.

## Reference Workspace

[`environments/gpt-world/`](/Users/alexandremaraval/Documents/Projects/prime-envs/environments/gpt-world) is the clearest current example of the root-level structure this repo is moving toward. Use it as the reference for how a self-contained environment workspace can be organized.
