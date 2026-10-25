from __future__ import annotations

from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
import run_e115_hierarchical_importance as runner


class E115ReadinessTests(unittest.TestCase):
    def test_direct_run_refuses_legacy_fit_before_data_reads_and_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'not_created'
            with patch.object(runner,'load_q2',side_effect=AssertionError('must not read training data')):
                with patch.object(runner,'_fit_backbone',side_effect=AssertionError('must not fit legacy ET')):
                    with self.assertRaisesRegex(RuntimeError,'final NNLS'):
                        runner.run_e115(repeat_ids=[0],output_dir=output)
            self.assertFalse(output.exists())

    def test_cli_execute_is_recognized_but_blocked_by_final_model_guard(self):
        errors=StringIO()
        with patch.object(sys,'argv',['run_e115','--execute']),redirect_stderr(errors):
            with patch.object(runner,'_fit_backbone',side_effect=AssertionError('must not fit')):
                with self.assertRaises(SystemExit) as raised:runner.main()
        self.assertEqual(raised.exception.code,2)
        self.assertIn('final NNLS',errors.getvalue())
        self.assertNotIn('unrecognized arguments',errors.getvalue())

    def test_default_cli_is_safe_readiness_report_not_execution(self):
        output=StringIO()
        with patch.object(sys,'argv',['run_e115']),redirect_stdout(output):
            with patch.object(runner,'run_e115',side_effect=AssertionError('default must not execute')):
                runner.main()
        data=json.loads(output.getvalue())
        self.assertTrue(data['adapter_ready'])
        self.assertFalse(data['E115_execution_ready'])
        self.assertFalse(data['q1_interface_verified'])

    def test_help_does_not_read_or_fit_data(self):
        with patch.object(sys,'argv',['run_e115','--help']),redirect_stdout(StringIO()):
            with patch.object(runner,'load_q2',side_effect=AssertionError('help must be read-free')):
                with self.assertRaises(SystemExit) as raised:runner.main()
        self.assertEqual(raised.exception.code,0)

if __name__=='__main__':unittest.main()
