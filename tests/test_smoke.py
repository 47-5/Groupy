import importlib.util
import builtins
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter
from groupy.gp_loader import Loader
from groupy.gp_tool import Tool
from groupy.io import load_smiles_file


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

    def test_counter_batch_uses_shared_smiles_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "count.csv"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")

            result = Counter().count_mols(
                str(input_path),
                count_result_file_path=str(output_path),
                add_smiles=True,
            )

            self.assertEqual(result.loc[0, "smiles"], "C1CCCC1")
            self.assertEqual(result.loc[0, "f_168"], 5)
            self.assertTrue(output_path.exists())

    def test_calculator_batch_uses_shared_smiles_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "calculate.csv"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")

            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                result = Calculator().calculate_mols(
                    str(input_path),
                    properties_file_path=str(output_path),
                )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result.loc[0, "smiles"], "C1CCCC1")
            self.assertAlmostEqual(result.loc[0, "molar_mass"], 70.135)
            self.assertTrue(output_path.exists())
            self.assertFalse((tmp_path / "error.txt").exists())

    def test_calculator_batch_writes_error_file_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "calculate.csv"
            error_path = tmp_path / "errors.txt"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")

            calculator = Calculator()

            def fail_calculation(*args, **kwargs):
                raise ValueError("forced failure")

            calculator.calculate_a_mol = fail_calculation
            result = calculator.calculate_mols(
                str(input_path),
                properties_file_path=str(output_path),
                error_file_path=str(error_path),
            )

            self.assertTrue(result.empty)
            self.assertEqual(error_path.read_text(encoding="utf-8"), "C1CCCC1\n")
            self.assertTrue(output_path.exists())

    def test_parallel_aliases_for_counter_and_calculator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            count_output = tmp_path / "count_parallel.csv"
            calculate_output = tmp_path / "calculate_parallel.csv"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")

            count_result = Counter().count_mols_parallel(
                str(input_path),
                count_result_file_path=str(count_output),
                add_smiles=True,
                n_jobs=1,
            )
            calculate_result = Calculator().calculate_mols_parallel(
                str(input_path),
                properties_file_path=str(calculate_output),
                n_jobs=1,
            )

            self.assertEqual(count_result.loc[0, "smiles"], "C1CCCC1")
            self.assertEqual(count_result.loc[0, "f_168"], 5)
            self.assertEqual(calculate_result.loc[0, "smiles"], "C1CCCC1")
            self.assertAlmostEqual(calculate_result.loc[0, "molar_mass"], 70.135)


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


class ConvertorSmokeTests(unittest.TestCase):
    def test_conversion_modules_do_not_import_openbabel_at_import_time(self):
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import groupy.gp_convertor; "
                "import groupy.gp_generator; "
                "print('openbabel' in sys.modules)"
            ),
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

    def test_lazy_pybel_import_reports_install_hint_when_missing(self):
        from groupy.gp_convertor import _load_pybel

        original_import = builtins.__import__

        def block_openbabel(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "openbabel":
                raise ImportError("blocked for test")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_openbabel
        try:
            with self.assertRaisesRegex(ImportError, "conda install -c conda-forge openbabel"):
                _load_pybel()
        finally:
            builtins.__import__ = original_import

    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by gp_convertor")
    def test_batch_smi_to_xyz_does_not_write_logs_by_default(self):
        from groupy.gp_convertor import Convertor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            xyz_root = tmp_path / "xyz"
            input_path.write_text("C1CCCC1\ninvalid\n", encoding="utf-8")

            convertor = Convertor()

            def fake_smi_to_xyz(smi, xyz_path=None):
                if smi == "invalid":
                    return False
                Path(xyz_path).write_text("fake xyz\n", encoding="utf-8")
                return True

            convertor.smi_to_xyz = fake_smi_to_xyz

            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                convertor.batch_smi_to_xyz(str(input_path), str(xyz_root))
            finally:
                os.chdir(original_cwd)

            self.assertFalse((tmp_path / "xyz_fail.txt").exists())
            self.assertFalse((tmp_path / "xyz_succeed.txt").exists())

    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by gp_convertor")
    def test_batch_smi_to_xyz_writes_logs_when_requested(self):
        from groupy.gp_convertor import Convertor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            xyz_root = tmp_path / "xyz"
            fail_path = tmp_path / "logs" / "xyz_fail.txt"
            succeed_path = tmp_path / "logs" / "xyz_succeed.txt"
            input_path.write_text("C1CCCC1\ninvalid\n", encoding="utf-8")

            convertor = Convertor()

            def fake_smi_to_xyz(smi, xyz_path=None):
                if smi == "invalid":
                    return False
                Path(xyz_path).write_text("fake xyz\n", encoding="utf-8")
                return True

            convertor.smi_to_xyz = fake_smi_to_xyz
            convertor.batch_smi_to_xyz(
                str(input_path),
                str(xyz_root),
                fail_file_path=str(fail_path),
                succeed_file_path=str(succeed_path),
            )

            self.assertEqual(fail_path.read_text(encoding="utf-8"), "invalid\n")
            self.assertEqual(succeed_path.read_text(encoding="utf-8"), "C1CCCC1\n")

    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by gp_convertor")
    def test_convertor_parallel_aliases_call_legacy_methods(self):
        from groupy.gp_convertor import Convertor

        convertor = Convertor()
        calls = []

        def fake_batch_smi_to_xyz_mpi(*args, **kwargs):
            calls.append(("xyz", args, kwargs))
            return [True]

        def fake_batch_convert_file_type_mpi(*args, **kwargs):
            calls.append(("convert", args, kwargs))
            return ["converted"]

        def fake_batch_file_to_smi_mpi(*args, **kwargs):
            calls.append(("smi", args, kwargs))
            return ["C1CCCC1"]

        convertor.batch_smi_to_xyz_mpi = fake_batch_smi_to_xyz_mpi
        convertor.batch_convert_file_type_mpi = fake_batch_convert_file_type_mpi
        convertor.batch_file_to_smi_mpi = fake_batch_file_to_smi_mpi

        self.assertEqual(
            convertor.batch_smi_to_xyz_parallel("smiles.txt", "xyz", n_jobs=1),
            [True],
        )
        self.assertEqual(
            convertor.batch_convert_file_type_parallel("xyz", "in", "mol2", n_jobs=1),
            ["converted"],
        )
        self.assertEqual(
            convertor.batch_file_to_smi_parallel("xyz", "in", n_jobs=1),
            ["C1CCCC1"],
        )
        self.assertEqual([call[0] for call in calls], ["xyz", "convert", "smi"])


class GeneratorSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by gp_generator")
    def test_batch_smi_to_gjf_does_not_write_logs_by_default(self):
        from groupy.gp_generator import Generator

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            gjf_root = tmp_path / "gjf"
            input_path.write_text("C1CCCC1\ninvalid\n", encoding="utf-8")

            generator = Generator()

            def fake_smi_to_gjf(smi, **kwargs):
                if smi == "invalid":
                    return False
                Path(kwargs["gjf_path"]).write_text("fake gjf\n", encoding="utf-8")
                return True

            generator.smi_to_gjf = fake_smi_to_gjf

            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                generator.batch_smi_to_gjf(str(input_path), str(gjf_root))
            finally:
                os.chdir(original_cwd)

            self.assertFalse((tmp_path / "gjf_fail.txt").exists())
            self.assertFalse((tmp_path / "gjf_succeed.txt").exists())

    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by gp_generator")
    def test_batch_smi_to_gjf_writes_logs_when_requested(self):
        from groupy.gp_generator import Generator

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            gjf_root = tmp_path / "gjf"
            fail_path = tmp_path / "logs" / "gjf_fail.txt"
            succeed_path = tmp_path / "logs" / "gjf_succeed.txt"
            input_path.write_text("C1CCCC1\ninvalid\n", encoding="utf-8")

            generator = Generator()

            def fake_smi_to_gjf(smi, **kwargs):
                if smi == "invalid":
                    return False
                Path(kwargs["gjf_path"]).write_text("fake gjf\n", encoding="utf-8")
                return True

            generator.smi_to_gjf = fake_smi_to_gjf
            generator.batch_smi_to_gjf(
                str(input_path),
                str(gjf_root),
                fail_file_path=str(fail_path),
                succeed_file_path=str(succeed_path),
            )

            self.assertEqual(fail_path.read_text(encoding="utf-8"), "invalid\n")
            self.assertEqual(succeed_path.read_text(encoding="utf-8"), "C1CCCC1\n")

    @unittest.skipUnless(importlib.util.find_spec("openbabel"), "OpenBabel is required by gp_generator")
    def test_generator_parallel_alias_calls_legacy_method(self):
        from groupy.gp_generator import Generator

        generator = Generator()
        calls = []

        def fake_batch_smi_to_gjf_mpi(*args, **kwargs):
            calls.append((args, kwargs))
            return [True]

        generator.batch_smi_to_gjf_mpi = fake_batch_smi_to_gjf_mpi

        self.assertEqual(
            generator.batch_smi_to_gjf_parallel("smiles.txt", "gjf", n_jobs=1),
            [True],
        )
        self.assertEqual(calls[0][0], ("smiles.txt", "gjf"))
        self.assertEqual(calls[0][1], {"n_jobs": 1})


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
