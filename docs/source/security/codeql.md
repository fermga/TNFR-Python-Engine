# CodeQL Analysis Workflow

The CodeQL workflow analyzes this repository for security vulnerabilities in the TNFR engine's Python code. The automation runs on GitHub Actions and is defined in `.github/workflows/codeql-analysis.yml`.

## When it runs

The analysis triggers automatically in the following situations:

- Pushes to the `main` or `master` branches.
- Pull requests that target `main` or `master`.
- A scheduled weekly execution (`cron`).

You can also start the workflow manually from the **Actions** tab by selecting _CodeQL Analysis_ and clicking **Run workflow**.

## What the workflow does

1. Checks out the repository.
2. Initializes CodeQL for the Python language.
3. Runs the `autobuild` step (no additional configuration is required for this project).
4. Analyzes the code and generates a SARIF report.
5. Uploads the results to GitHub Advanced Security and stores them as a run artifact.

## How to review findings

1. Open the repository's **Security** tab in GitHub.
2. Select **Code scanning alerts** to see the complete list of findings. You can filter by status, severity, or tool.
3. Open an alert to review the file, precise location, and trace that triggered the detection.
4. Mark the alert as resolved or `won't fix` as appropriate, documenting the decision in a comment.

## GitHub Security dashboard

- In **Security > Overview** you will find aggregated metrics, trends, and shortcuts to the most relevant alerts.
- The execution history for CodeQL lives under **Security > Code scanning**. Each run links to the uploaded artifact and processed SARIF.
- To download the results artifact, open the run in **Actions**, expand the _Analyze_ job, and download `codeql-python-results`.

Keep the workflow active and review alerts regularly to preserve the structural coherence and security of the TNFR engine.
