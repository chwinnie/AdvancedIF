# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from AdvancedIF.cli import cli


class TestCLIEvaluate(unittest.TestCase):
    """Test cases for CLI evaluate command."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("AdvancedIF.cli.IFRubricsJudge")
    @patch("AdvancedIF.cli.SystemSteerIFRubricsJudge")
    @patch("AdvancedIF.cli.DataProcessor")
    def test_evaluate_runs_with_required_args_only(
        self,
        mock_processor: MagicMock,
        mock_system_judge: MagicMock,
        mock_if_judge: MagicMock,
    ) -> None:
        """Test evaluate runs with only input, output, api-key, model."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_file.return_value = {
            "total_rows": 1,
            "filtered_rows": 0,
            "processed_rows": 1,
            "successful_rows": 1,
            "failed_rows": 0,
            "success_rate": 1.0,
            "overall_pass_rate": 1.0,
            "micro_pass_rate": 1.0,
            "passed_rubrics": 1,
            "total_rubrics": 1,
        }
        mock_processor.return_value = mock_processor_instance

        with self.runner.isolated_filesystem():
            with open("input.csv", "w") as f:
                f.write("response,conversation_history,prompt_metadata,benchmark_name\n")
                f.write('"test","[]","{}","test"\n')

            result = self.runner.invoke(
                cli,
                [
                    "evaluate",
                    "--input", "input.csv",
                    "--output", "output.csv",
                    "--api-key", "test-key",
                    "--model", "o3-mini-2025-01-31",
                ],
            )

            self.assertEqual(result.exit_code, 0, f"CLI failed with: {result.output}")


if __name__ == "__main__":
    unittest.main()
