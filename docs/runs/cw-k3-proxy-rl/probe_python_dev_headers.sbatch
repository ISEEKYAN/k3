#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=k3-python-dev-probe
#SBATCH --output=k3-python-dev-probe-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

srun \
  --account=coreai_devtech_all \
  --container-image="${K3_TRAINING_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
/usr/bin/python3 - <<'"'"'PY'"'"'
import json
import sysconfig

print(
    "K3_PYTHON_SYSCONFIG="
    + json.dumps(
        {
            "paths": sysconfig.get_paths(),
            "multiarch": sysconfig.get_config_var("MULTIARCH"),
            "include_py": sysconfig.get_config_var("INCLUDEPY"),
            "confincludedir": sysconfig.get_config_var("CONFINCLUDEDIR"),
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
find /usr/include /usr/local/include -type f \
  \( -name Python.h -o -name pyconfig.h \) -print 2>/dev/null \
  | sort
'
