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

## Contribution Checklist

1. Pull the latest `main` branch and create a named feature branch.
2. Link the branch and pull request to an issue.
3. Run the relevant checks locally and describe the results in the pull request.
4. Address review feedback, obtain approval, and merge only after checks pass.

## Issue and Pull Request Quality

Issue titles use an action verb and identify the outcome, for example, "Create data quality report for incoming datasets". Each issue explains why the work matters and defines what done means. Pull requests summarize the implementation, testing, and related issue so a reviewer can evaluate the change in context.