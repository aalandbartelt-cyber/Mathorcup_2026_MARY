"""Read-only E118 input check; this script never generates a ranking."""

import json

from downstream_input_gates import audit_e118_inputs
from q2_contract import RESULT_DIR


if __name__ == "__main__":
    print(json.dumps(audit_e118_inputs(RESULT_DIR), ensure_ascii=False, indent=2))
