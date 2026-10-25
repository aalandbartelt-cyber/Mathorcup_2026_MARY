"""Read-only E119 input check; this script never grades or reranks entries."""

import json

from downstream_input_gates import audit_e119_inputs
from q2_contract import RESULT_DIR


if __name__ == "__main__":
    print(json.dumps(audit_e119_inputs(RESULT_DIR), ensure_ascii=False, indent=2))
