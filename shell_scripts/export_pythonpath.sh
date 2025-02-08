repo_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$(dirname $repo_root):$repo_root/llmtf_open:$PYTHONPATH
