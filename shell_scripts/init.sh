repo_root=$(git rev-parse --show-toplevel)
git submodule update --init --recursive
source $repo_root/shell_scripts/export_pythonpath.sh
