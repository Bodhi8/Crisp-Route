# GitHub Upload Checklist

Target repository: `https://github.com/Bodhi8/crisp-route`

```bash
git init
git add .
git commit -m "Initial CRISP-Route release"
git branch -M main
git remote add origin https://github.com/Bodhi8/crisp-route.git
git push -u origin main
```

Before publishing:

1. Create the empty `crisp-route` repository under `Bodhi8` without adding a README or license.
2. Run the test and benchmark commands in `README.md`.
3. Confirm the GitHub Actions workflow passes on Python 3.10–3.12.
4. Add repository topics: `routing`, `shortest-path`, `multi-objective`, `cvar`, `dynamic-graphs`, `operations-research`, `python`.
5. Create a `v0.1.0` release after the initial commit is public.
