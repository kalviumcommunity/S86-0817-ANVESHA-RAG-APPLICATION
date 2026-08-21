# Team GitHub Workflow

## Branching Strategy

- `main` contains releasable code only.
- New work starts from `main` in a branch named `feature/[description]`.
- Bug fixes use `fix/[description]`; documentation uses `docs/[description]`.
- Branches are deleted after their pull requests are merged.

## Commit Message Convention

Commit messages use the format `[type]: [description]`.

The team uses these types:

- `feat`: add a capability
- `fix`: correct broken behavior
- `docs`: update documentation
- `refactor`: improve structure without changing behavior
- `chore`: maintain tooling or project configuration

Consistent messages make the history clear and enable automated changelog generation.

## Pull Request Review Process

- Every pull request requires at least one approval before merge.
- Reviewers check correctness, clarity, data integrity, and test coverage.
- Commit messages are reviewed as part of the code review.
- Pull requests link the issue they resolve with `Closes #[issue-number]` or `Fixes #[issue-number]`.

## GitHub Issue Tracking

- Every feature or fix starts with an issue.
- Issues include an action-oriented title, context, acceptance criteria, labels, and an assignee.
- Issues are closed when the corresponding pull request is merged.