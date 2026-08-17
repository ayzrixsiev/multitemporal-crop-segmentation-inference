# Deploying CropScope to Hugging Face Spaces

Free CPU tier, Docker SDK, container on port 7860.

This repo is **not** the Space repo. `.gitignore` here excludes `results/`,
`/samples/`, `*.pth.tar`, `*.pkl` and `*.npz`, so pushing it straight to a Space
would ship an app with no model and no patches. `deploy/build_space.sh` assembles
a separate staging directory holding exactly what the server reads at runtime;
that directory is what gets pushed.

Nothing below modifies this repo.

---

## One-time setup

**1. Create the Space.** On <https://huggingface.co/new-space>: pick a name,
select **Docker** as the SDK (blank template), hardware **CPU basic — free**,
visibility **Public**. Leave it empty; the README that configures it is pushed
in step 5.

**2. Install git-lfs.** The checkpoint (13 MB) and the ten `.npz` bundles
(9–15 MB each) are over Hugging Face's 10 MB plain-git limit and must go through
LFS. `deploy/.gitattributes` already declares the two patterns.

```bash
git lfs install
```

**3. Clone the empty Space** somewhere outside this repo.

```bash
git clone https://huggingface.co/spaces/<user>/<space-name> ~/cropscope-space
```

---

## Every deploy

**4. Build the staging directory** from the repo root. It is wiped and rebuilt
each run, and prints the staged size (~120 MB).

```bash
bash deploy/build_space.sh
```

**5. Copy it into the clone.** `--delete` makes the Space match the staging
directory exactly, so a file removed here is removed there too; `--exclude .git`
keeps rsync away from the clone's own git metadata.

```bash
rsync -a --delete --exclude .git deploy/space-build/ ~/cropscope-space/
```

**6. Commit and push.** Git-lfs picks up the checkpoint and the bundles from the
`.gitattributes` copied in at step 5.

```bash
cd ~/cropscope-space
git add -A
git commit -m "Deploy CropScope"
git push
```

When git asks for credentials, the username is your Hugging Face username and
the password is a **write** access token from
<https://huggingface.co/settings/tokens> — your account password will not work.
To avoid retyping it, run `git config credential.helper store` in the clone
before pushing.

**7. Watch the build.** The Space page shows the Docker build log. First build
takes a few minutes, most of it installing torch. Once it says *Running*, the
app is live at `https://huggingface.co/spaces/<user>/<space-name>`.

---

## Redeploying after a code change

Edit the code here, then repeat steps 4–6. Only what actually changed is
re-uploaded, and the image layers are cached: because `requirements.txt` is
copied and installed before the app, a change to `webapp/` or `src/` does not
reinstall torch.

If a change touches `deploy/Dockerfile`, `deploy/requirements.txt`,
`deploy/.gitattributes` or `deploy/space_README.md`, the rebuild picks it up the
same way — those four are copied into the staging directory by the build script.

---

## What to expect from the free tier

- **A free Space sleeps after about 48 hours without traffic**, and takes
  roughly 30–60 seconds to wake on the next visit. Open the link once shortly
  before sharing it so the first person to click gets a running app.
- 2 vCPU, 16 GB RAM. One patch takes a few seconds to run — around 4 s for a
  38-date patch and around 6 s for the 61-date one, measured on a CPU-capped
  container.
- Only `/home/user` and `/tmp` are writable, and the container runs as uid 1000.
  The Dockerfile already matches that; do not move the app elsewhere.

## Notes

- `deploy/space-build/` is untracked here and most of it is already covered by
  `.gitignore` patterns. It is a build artifact — do not commit it to this repo.
- The Space serves `samples/group1/` only. `group2/` (111 MB) is deliberately
  left out: the frontend does not show it.
