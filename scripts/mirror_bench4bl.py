"""Downloads and extracts Bench4BL project archives from SourceForge into bench4bl_cache/,
ready for dataset/bench4bl.py to read directly -- see docs/next_steps.md for the scoping
notes this is based on. Each archive already contains a pre-generated bug-repository XML
and a real git repo, so no legacy Python 2 tooling is needed at any point in this pipeline.

Usage:
    python scripts/mirror_bench4bl.py --projects WEAVER CRYPTO IO CODEC
    python scripts/mirror_bench4bl.py --all
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects', nargs='+', choices=list(PROJECT_GROUPS), default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

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
