# CI performance and coverage

The `Tests` workflow keeps its existing three check names. Python, Java and Vue
run in parallel after a small changed-file check.

- PR updates cancel older runs for the same PR. Each non-PR run has a unique
  concurrency group, so every main commit can finish, including pending runs.
- Documentation-only PRs skip the three expensive jobs. Their check names remain
  present with a skipped result, which GitHub accepts for required checks.
  All main commits run all three jobs, including documentation changes.
- Markdown and documentation files are documentation, except `docs/docker/`,
  which contains runtime files. Renames also inspect the previous path.
  Incomplete changed-file listings run the checks. A failed file check fails
  each existing check name before dependency setup.
- Python installs the complete `requirements.txt` with the precompiled `uv`
  installer. CPU-only GitHub runners use PyTorch and torchaudio CPU wheels at
  the same pinned versions. Import checks confirm both wheels load.
- `setup-uv` manages and prunes its dependency cache. Do not also cache pip or a
  full virtual environment. The old pip cache stored about 3.19 GB per PR.
- Java uses `setup-java` Maven caching. Vue uses `setup-node` npm caching.
  The npm lockfile keeps package versions and integrity hashes while using the
  official npm registry. Installation prefers cached package downloads.
- Dependency installation and Vue tests have separate timing steps. The suites
  take only seconds, so sharding would add more setup than it saves.

## Existing coverage

Python runs the existing two smoke tests, plus the CPU wheel import check. Vue
runs the existing 70 unit tests. No existing check or test was removed.

The existing Java command runs Maven's test lifecycle. The POM sets
`skipTests=true`, so this currently compiles main and test sources but does not
execute Java tests. Some tests require a full Spring application and a local
Redis service. This performance change does not supply production credentials,
change those tests, or silently claim that they execute. Enabling their complete
test environment is separate work.

## Reproduce the measurements

Use `gh run list --repo PG2047/xiaozhi-esp32-server --limit 50` and the Actions
jobs API for queue, setup and work times. Before this change only 38 runs existed.
The Python median job took 177 seconds, with 176 seconds outside its one-second
smoke step. Vue's combined install/test step hid a median 31.13 seconds of npm
installation and about 2.00 seconds of tests. Java's historical work duration
includes dependency resolution and compilation, not test execution.

Run `npm ci --registry=https://registry.npmjs.org --prefer-offline --no-audit
--no-fund`, then `npm run build` twice from `main/manager-web`. On the validation
Mac the two builds took 13.62 seconds and 3.16 seconds. The existing Webpack disk
cache works; no application build behavior needed to change.

Docker release workflows publish images and are not triggered for performance
experiments. Their first-stage Dockerfiles already copy package manifests before
application source. No release or production deployment is required to validate
these CI changes.
