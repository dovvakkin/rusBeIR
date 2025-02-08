repo_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$repo_root:$repo_root/llmtf_open:$PYTHONPATH
