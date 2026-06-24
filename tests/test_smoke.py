import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from groupy.api import load_smiles_file
from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter
from groupy.gp_loader import Loader
from groupy.gp_tool import Tool


class LoaderSmokeTests(unittest.TestCase):
    def test_loader_reads_bundled_data(self):
        loader = Loader()

        parameters = loader.load_parameters()
        group_orders = loader.load_group_order()

        self.assertEqual(len(parameters), 425)
        self.assertEqual(group_orders[0][:3], [104, 105, 93])
        self.assertTrue(all(group_orders))


class CoreChemistrySmokeTests(unittest.TestCase):
    def test_counter_counts_cyclopentane(self):
        result = Counter().count_a_mol("C1CCCC1", clear_mode=True, add_smiles=True)

        self.assertEqual(result, {"f_168": 5, "smiles": "C1CCCC1"})

    def test_calculator_calculates_cyclopentane(self):
        result = Calculator().calculate_a_mol("C1CCCC1")

        self.assertEqual(result["smiles"], "C1CCCC1")
        self.assertAlmostEqual(result["molar_mass"], 70.135)
        self.assertAlmostEqual(result["Tb/K"], 308.65)
        self.assertAlmostEqual(result["Pc/bar"], 42.659)
        self.assertEqual(result["note"], "C1CCCC1 at 298K")


class SmilesFileSmokeTests(unittest.TestCase):
    def test_load_smiles_iterator_from_txt_csv_and_xlsx(self):
        expected = ["C1CCCC1", "CCO"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            txt_path = tmp_path / "smiles.txt"
            csv_path = tmp_path / "smiles.csv"
            xlsx_path = tmp_path / "smiles.xlsx"

            txt_path.write_text("C1CCCC1\nCCO\n", encoding="utf-8")
            pd.DataFrame({"smiles": expected}).to_csv(csv_path, index=False)
            pd.DataFrame({"smiles": expected}).to_excel(xlsx_path, index=False)

            self.assertEqual(Tool.load_smiles_iterator(str(txt_path)), expected)
            self.assertEqual(Tool.load_smiles_iterator(str(csv_path)), expected)
            self.assertEqual(Tool.load_smiles_iterator(str(xlsx_path)), expected)
            self.assertEqual(load_smiles_file(txt_path), expected)
            self.assertEqual(load_smiles_file(csv_path), expected)
            self.assertEqual(load_smiles_file(xlsx_path), expected)


class CliSmokeTests(unittest.TestCase):
    def test_cli_import_does_not_load_openbabel_conversion_stack(self):
        command = [
            sys.executable,
            "-c",
            "import sys; import groupy.cli; print('groupy.gp_convertor' in sys.modules)",
        ]

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")

    def test_count_cli_outputs_json(self):
        command = [
            sys.executable,
            "-m",
            "groupy.cli",
            "count",
            "--smiles",
            "C1CCCC1",
        ]

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), {"f_168": 5, "smiles": "C1CCCC1"})

    def test_count_cli_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "count.csv"
            command = [
                sys.executable,
                "-m",
                "groupy.cli",
                "count",
                "--smiles",
                "C1CCCC1",
                "--output",
                str(output_path),
            ]

            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = pd.read_csv(output_path)
            self.assertEqual(result.to_dict(orient="records"), [{"f_168": 5, "smiles": "C1CCCC1"}])

    def test_calculate_cli_outputs_json(self):
        command = [
            sys.executable,
            "-m",
            "groupy.cli",
            "calculate",
            "--smiles",
            "C1CCCC1",
        ]

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["smiles"], "C1CCCC1")
        self.assertAlmostEqual(result["molar_mass"], 70.135)
        self.assertAlmostEqual(result["Tb/K"], 308.65)
        self.assertEqual(result["note"], "C1CCCC1 at 298K")

    def test_calculate_cli_writes_csv_from_input_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "calculate.csv"
            input_path.write_text("C1CCCC1\nCCO\n", encoding="utf-8")

            command = [
                sys.executable,
                "-m",
                "groupy.cli",
                "calculate",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]

            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = pd.read_csv(output_path)
            self.assertEqual(result["smiles"].tolist(), ["C1CCCC1", "CCO"])
            self.assertIn("molar_mass", result.columns)

    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by the legacy interactive CLI")
    def test_legacy_cli_can_start_and_exit(self):
        command = [
            sys.executable,
            "-m",
            "groupy.cli",
            "interactive",
        ]

        completed = subprocess.run(
            command,
            input="q\n",
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("exit Groupy", completed.stdout)


if __name__ == "__main__":
    unittest.main()
