"""Downloads and extracts Bench4BL project archives from SourceForge into bench4bl_cache/,
ready for dataset/bench4bl.py to read directly -- see docs/next_steps.md for the scoping
notes this is based on. Each archive already contains a pre-generated bug-repository XML
and a real git repo, so no legacy Python 2 tooling is needed at any point in this pipeline.

--iqloc additionally prepares bench4bl_cache/ for dataset/iqloc.py's IQLocExtended loader,
which reads Bench4BLExtended.json directly instead of the SourceForge archives' bugrepo XML.
Bench4BLExtended.json's `version` field is the literal git tag (unlike Bench4BL's own
versions.txt-indexed labels), so IQLoc's ~30% newer bug reports need two things the plain
archives don't provide: (1) tags released after each archive's SourceForge snapshot, fetched
straight from the project's live GitHub origin, and (2) full clones for sub-projects that
split out of an Apache monorepo (camel-karaf/-quarkus/-spring-boot) or were renamed
(commons-compress/-csv/-lang/-io) after the archive was built, which have no SourceForge
archive at all.

Usage:
    python scripts/mirror_bench4bl.py --projects WEAVER CRYPTO IO CODEC
    python scripts/mirror_bench4bl.py --all
    python scripts/mirror_bench4bl.py --iqloc
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import get_logger, setup_logging
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# project -> SourceForge group, scraped from github.com/exatoa/Bench4BL's downloads.sh
PROJECT_GROUPS = {
    "CAMEL": "Apache", "HBASE": "Apache", "HIVE": "Apache",
    "CODEC": "Commons", "COLLECTIONS": "Commons", "COMPRESS": "Commons",
    "CONFIGURATION": "Commons", "CRYPTO": "Commons", "CSV": "Commons",
    "IO": "Commons", "LANG": "Commons", "MATH": "Commons", "WEAVER": "Commons",
    "ENTESB": "JBoss", "JBMETA": "JBoss",
    "ELY": "Wildfly", "SWARM": "Wildfly", "WFARQ": "Wildfly", "WFCORE": "Wildfly",
    "WFLY": "Wildfly", "WFMP": "Wildfly",
    "AMQP": "Spring", "ANDROID": "Spring", "BATCH": "Spring", "BATCHADM": "Spring",
    "DATACMNS": "Spring", "DATAGRAPH": "Spring", "DATAJPA": "Spring", "DATAMONGO": "Spring",
    "DATAREDIS": "Spring", "DATAREST": "Spring", "LDAP": "Spring", "MOBILE": "Spring",
    "ROO": "Spring", "SEC": "Spring", "SECOAUTH": "Spring", "SGF": "Spring",
    "SHDP": "Spring", "SHL": "Spring", "SOCIAL": "Spring", "SOCIALFB": "Spring",
    "SOCIALLI": "Spring", "SOCIALTW": "Spring", "SPR": "Spring", "SWF": "Spring",
    "SWS": "Spring",
    "JDT": "Previous", "PDE": "Previous", "SWT": "Previous",
}

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench4bl_cache")

# sub_projects in Bench4BLExtended.json with no SourceForge archive at all -- confirmed by
# running scripts/smoke_test_iqloc_loader.py and checking each github.com/apache/<name> URL
# resolves. Cloned directly into bench4bl_cache/<name>/gitrepo (no bugrepo/versions.txt
# needed here since IQLocExtended reads bug metadata from the JSON, not the archive's XML).
IQLOC_EXTRA_REPOS = {
    "commons-compress": "apache/commons-compress",
    "commons-csv": "apache/commons-csv",
    "commons-lang": "apache/commons-lang",
    "commons-io": "apache/commons-io",
    "camel-karaf": "apache/camel-karaf",
    "camel-quarkus": "apache/camel-quarkus",
    "camel-spring-boot": "apache/camel-spring-boot",
}


def _download_and_extract(project, group, cache_dir):
    url = f"https://sourceforge.net/projects/irblsensitivity/files/{group}/{project}.tar/download"
    dest = os.path.join(cache_dir, project)
    if os.path.isdir(os.path.join(dest, "bugrepo")):
        logger.info(f"{project}: already mirrored, skipping (delete {dest} to re-fetch)")
        return

    logger.info(f"{project}: downloading from {url}")
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        urllib.request.urlretrieve(url, tmp_path)
        size_mb = os.path.getsize(tmp_path) / 1e6
        logger.info(f"{project}: downloaded {size_mb:.1f}MB, extracting")

        with tempfile.TemporaryDirectory() as extract_tmp:
            with tarfile.open(tmp_path) as tf:
                tf.extractall(extract_tmp)
            os.makedirs(dest, exist_ok=True)
            for sub in ("bugrepo", "gitrepo", "versions.txt"):
                src = os.path.join(extract_tmp, sub)
                if os.path.exists(src):
                    target = os.path.join(dest, sub)
                    if os.path.isdir(src):
                        shutil.copytree(src, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, target)
                else:
                    logger.warning(f"{project}: archive missing expected member {sub}")
        logger.info(f"{project}: mirrored to {dest}")
    finally:
        os.unlink(tmp_path)


def _refresh_tags(project, cache_dir):
    """Pulls newer tags into an already-mirrored project's gitrepo from its live GitHub
    origin. Needed because Bench4BLExtended.json's bug reports reference versions released
    after the SourceForge archive was snapshotted -- the archive's own gitrepo has no way to
    get those without talking to the real upstream."""
    gitrepo = os.path.join(cache_dir, project, "gitrepo")
    if not os.path.isdir(gitrepo):
        logger.warning(f"{project}: no gitrepo at {gitrepo}, skipping tag refresh (mirror it first)")
        return
    logger.info(f"{project}: fetching tags from origin")
    subprocess.run(["git", "-C", gitrepo, "fetch", "--tags", "--force", "origin"], check=True)


def _clone_extra_repo(name, github_path, cache_dir):
    dest = os.path.join(cache_dir, name, "gitrepo")
    if os.path.isdir(dest):
        logger.info(f"{name}: already cloned, skipping (delete {dest} to re-clone)")
        return
    logger.info(f"{name}: cloning https://github.com/{github_path}.git")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", f"https://github.com/{github_path}.git", dest], check=True)


def _needed_tags_for_iqloc(dataset_json_path):
    """Reads Bench4BLExtended.json and returns {sub_project: [tags]} -- exactly the git refs
    IQLoc's dataset actually references, nothing more. Used to build minimal bundles instead
    of transferring a repo's entire history/branches/reflogs (see the CAMEL reflog-bloat
    incident: 186,914 files to check for one project's tag refresh alone, almost all of it
    dependabot-branch reflog noise nothing here needs)."""
    with open(dataset_json_path) as f:
        records = json.load(f)
    tags_by_project = {}
    for rec in records:
        tags_by_project.setdefault(rec["sub_project"], set()).add(rec["version"])
    return {project: sorted(tags) for project, tags in tags_by_project.items()}


def _create_bundle(project, cache_dir, tags, bundle_dir):
    """git bundle create only pulls in objects reachable from the given refs -- no
    dependabot branches, no reflogs, no unreferenced history. Verified immediately after
    creation (git bundle verify) so a corrupt/incomplete bundle fails loudly here, not
    silently after transfer to MN5."""
    gitrepo = os.path.join(cache_dir, project, "gitrepo")
    if not os.path.isdir(gitrepo):
        logger.warning(f"{project}: no gitrepo at {gitrepo}, skipping bundle (mirror it first)")
        return None
    existing_tags = set(subprocess.run(
        ["git", "-C", gitrepo, "tag"], capture_output=True, text=True, check=True,
    ).stdout.splitlines())
    usable_tags = [t for t in tags if t in existing_tags]
    missing = set(tags) - existing_tags
    if missing:
        logger.warning(f"{project}: {len(missing)} referenced tag(s) not found locally, skipping those: {sorted(missing)[:5]}")
    if not usable_tags:
        logger.warning(f"{project}: no usable tags, skipping bundle")
        return None

    os.makedirs(bundle_dir, exist_ok=True)
    bundle_path = os.path.join(bundle_dir, f"{project}.bundle")
    subprocess.run(
        ["git", "-C", gitrepo, "bundle", "create", os.path.abspath(bundle_path), *usable_tags],
        check=True,
    )
    subprocess.run(["git", "bundle", "verify", bundle_path], check=True, capture_output=True)
    size_mb = os.path.getsize(bundle_path) / 1e6
    logger.info(f"{project}: bundle created and verified, {len(usable_tags)} tags, {size_mb:.1f}MB -> {bundle_path}")
    return bundle_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects', nargs='+', default=None,
                         help="Restrict to these projects. For --bundle, any bench4bl_cache/ "
                              "subdirectory name is accepted (including IQLOC_EXTRA_REPOS); "
                              "for the SourceForge download modes, must be a PROJECT_GROUPS key.")
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    parser.add_argument('--bundle', action='store_true',
                         help="Create minimal git bundles (only tags Bench4BLExtended.json "
                              "actually references) instead of transferring full clones. Use "
                              "for MN5 transfers going forward -- avoids the reflog/branch "
                              "bloat a plain rsync of bench4bl_cache/ hits on large monorepos "
                              "like CAMEL. Import on MN5 with: "
                              "git clone --bare <project>.bundle <project>/gitrepo.git")
    parser.add_argument('--bundle-dir', default=os.path.join(
        os.path.dirname(DEFAULT_CACHE_DIR), "bundles"))
    parser.add_argument('--dataset-json', default=os.path.join(DEFAULT_CACHE_DIR, "Bench4BLExtended.json"),
                         help="With --bundle: source of the needed-tags list.")
    parser.add_argument('--iqloc', action='store_true',
                         help="Refresh tags on already-mirrored projects (or those given via "
                              "--projects) and clone the sub-projects Bench4BLExtended.json "
                              "needs that have no SourceForge archive. Prints resolution stats "
                              "afterward -- run scripts/smoke_test_iqloc_loader.py for the full "
                              "breakdown.")
    args = parser.parse_args()

    if args.iqloc:
        os.makedirs(args.cache_dir, exist_ok=True)
        refresh_projects = args.projects or sorted(PROJECT_GROUPS)
        for project in refresh_projects:
            try:
                _refresh_tags(project, args.cache_dir)
            except Exception as e:
                logger.error(f"{project}: tag refresh failed -- {e}")
        for name, github_path in IQLOC_EXTRA_REPOS.items():
            try:
                _clone_extra_repo(name, github_path, args.cache_dir)
            except Exception as e:
                logger.error(f"{name}: clone failed -- {e}")
        return

    if args.bundle:
        tags_by_project = _needed_tags_for_iqloc(args.dataset_json)
        bundle_projects = args.projects or sorted(tags_by_project)
        total_bytes = 0
        for project in bundle_projects:
            tags = tags_by_project.get(project)
            if not tags:
                logger.warning(f"{project}: not referenced by {args.dataset_json}, skipping")
                continue
            try:
                path = _create_bundle(project, args.cache_dir, tags, args.bundle_dir)
                if path:
                    total_bytes += os.path.getsize(path)
            except Exception as e:
                logger.error(f"{project}: bundle failed -- {e}")
        logger.info(f"Total bundle size: {total_bytes / 1e6:.1f}MB across {len(bundle_projects)} project(s), in {args.bundle_dir}")
        return

    if args.all:
        projects = sorted(PROJECT_GROUPS)
    elif args.projects:
        projects = args.projects
    else:
        parser.error("pass --projects <NAMES...> or --all")

    os.makedirs(args.cache_dir, exist_ok=True)
    for project in projects:
        try:
            _download_and_extract(project, PROJECT_GROUPS[project], args.cache_dir)
        except Exception as e:
            logger.error(f"{project}: failed -- {e}")


if __name__ == "__main__":
    main()
