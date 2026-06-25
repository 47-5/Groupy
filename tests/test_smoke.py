import importlib.util
import builtins
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter
from groupy.gp_loader import Loader
from groupy.gp_tool import Tool
from groupy.io import load_smiles_file


class LoaderSmokeTests(unittest.TestCase):
    def tearDown(self):
        Loader.clear_cache()

    def test_loader_reads_bundled_data(self):
        loader = Loader()

        parameters = loader.load_parameters()
        group_orders = loader.load_group_order()

        self.assertEqual(len(parameters), 425)
        self.assertEqual(group_orders[0][:3], [104, 105, 93])
        self.assertTrue(all(group_orders))

    def test_loader_caches_bundled_excel_reads(self):
        Loader.clear_cache()
        read_calls = []
        original_read_excel = Loader._read_excel

        def counting_read_excel(data_file, **kwargs):
            read_calls.append(kwargs["sheet_name"])
            return original_read_excel(data_file, **kwargs)

        Loader._read_excel = staticmethod(counting_read_excel)
        try:
            loader = Loader()
            loader.load_parameters(parameter_type="simultaneous")
            Loader().load_parameters(parameter_type="simultaneous")
            loader.load_group_order()
            Loader().load_group_order()
        finally:
            Loader._read_excel = staticmethod(original_read_excel)

        self.assertEqual(
            read_calls,
            [
                "simultaneous_first_order",
                "simultaneous_second_order",
                "simultaneous_third_order",
                "simultaneous_constants",
                "f",
                "s",
                "t",
            ],
        )

    def test_loader_returns_independent_copies_from_cache(self):
        Loader.clear_cache()
        loader = Loader()

        parameters = loader.load_parameters()
        parameter_key = next(iter(parameters))
        property_key = next(iter(parameters[parameter_key]))
        original_value = parameters[parameter_key][property_key]
        parameters[parameter_key][property_key] = "changed"

        group_orders = loader.load_group_order()
        group_orders[0].append(999999)

        fresh_parameters = Loader().load_parameters()
        fresh_group_orders = Loader().load_group_order()

        self.assertEqual(fresh_parameters[parameter_key][property_key], original_value)
        self.assertNotIn(999999, fresh_group_orders[0])


class PackagingSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tomllib"), "tomllib is available on Python 3.11+")
    def test_optional_dependency_groups_keep_viewer_out_of_core(self):
        import tomllib

        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]

        dependencies = project["dependencies"]
        optional_dependencies = project["optional-dependencies"]

        self.assertNotIn("ase", dependencies)
        self.assertEqual(optional_dependencies["viewer"], ["ase"])
        self.assertEqual(optional_dependencies["gui"], ["PySide6"])
        self.assertEqual(optional_dependencies["package"], ["PyInstaller"])
        self.assertEqual(optional_dependencies["convert"], [])
        self.assertIn("pytest", optional_dependencies["dev"])

    def test_windows_package_script_dry_run_uses_gui_entry(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_windows_app.py"
        completed = subprocess.run(
            [sys.executable, str(script_path), "--dry-run"],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PyInstaller command:", completed.stdout)
        self.assertIn("--windowed", completed.stdout)
        self.assertIn("--onedir", completed.stdout)
        self.assertIn("--collect-data=groupy", completed.stdout)
        self.assertIn("--collect-submodules=rdkit", completed.stdout)
        self.assertIn("--exclude-module=matplotlib", completed.stdout)
        self.assertIn("--exclude-module=IPython", completed.stdout)
        self.assertIn("groupy_gui_entry.py", completed.stdout)
        self.assertIn("dist", completed.stdout)

    def test_windows_package_script_accepts_custom_excludes(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_windows_app.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--dry-run",
                "--no-default-excludes",
                "--exclude-module",
                "example_unused_module",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--exclude-module=example_unused_module", completed.stdout)
        self.assertNotIn("--exclude-module=matplotlib", completed.stdout)

    def test_readme_documents_packaged_app_limitations(self):
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme = readme_path.read_text(encoding="utf-8")

        self.assertIn("OpenBabel from conda-forge", readme)
        self.assertIn("not part of the default GUI workflow", readme)
        self.assertIn("clean Windows machine", readme)
        self.assertIn("Package size optimization is intentionally deferred", readme)
        self.assertIn("RELEASE_CHECKLIST.md", readme)

    def test_release_checklist_documents_double_click_distribution(self):
        checklist_path = Path(__file__).resolve().parents[1] / "RELEASE_CHECKLIST.md"
        checklist = checklist_path.read_text(encoding="utf-8")

        self.assertIn("dist/Groupy/Groupy.exe", checklist)
        self.assertIn("dist/Groupy", checklist)
        self.assertIn("_internal", checklist)
        self.assertIn("clean Windows", checklist)
        self.assertIn("C1CCCC1", checklist)
        self.assertIn("CSV export", checklist)
        self.assertIn("OpenBabel from conda-forge", checklist)


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

    def test_calculator_reports_invalid_smiles(self):
        with self.assertLogs("groupy.gp_calculator", level="WARNING") as log_context:
            result = Calculator().calculate_a_mol("not-a-smiles")

        self.assertEqual(result["smiles"], "not-a-smiles")
        self.assertEqual(result["molar_mass"], "?")
        self.assertIn("Invalid SMILES", result["error"])
        self.assertIn("Failed to calculate properties", log_context.output[0])

    def test_calculator_rejects_unknown_parameter_type(self):
        with self.assertRaisesRegex(ValueError, "parameter_type"):
            Calculator().calculate_a_mol("C1CCCC1", parameter_type="unknown")

    def test_counter_logs_invalid_smiles(self):
        counter = Counter()

        with self.assertLogs("groupy.gp_counter", level="WARNING") as log_context:
            result = counter.count_a_mol("not-a-smiles")

        self.assertEqual(result, counter.init_result)
        self.assertIn("Failed to count groups", log_context.output[0])

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

    def test_counter_batch_can_run_quietly_for_gui_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "count.csv"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = Counter().count_mols(
                    str(input_path),
                    count_result_file_path=str(output_path),
                    add_smiles=True,
                    verbose=False,
                )

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(result.loc[0, "smiles"], "C1CCCC1")
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

    def test_calculator_batch_can_run_quietly_for_gui_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "calculate.csv"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = Calculator().calculate_mols(
                    str(input_path),
                    properties_file_path=str(output_path),
                    verbose=False,
                )

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(result.loc[0, "smiles"], "C1CCCC1")
            self.assertAlmostEqual(result.loc[0, "molar_mass"], 70.135)
            self.assertTrue(output_path.exists())

    def test_calculator_batch_writes_error_file_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            output_path = tmp_path / "calculate.csv"
            error_path = tmp_path / "errors.txt"
            input_path.write_text("not-a-smiles\n", encoding="utf-8")

            with self.assertLogs("groupy.gp_calculator", level="WARNING"):
                result = Calculator().calculate_mols(
                    str(input_path),
                    properties_file_path=str(output_path),
                    error_file_path=str(error_path),
                )

            self.assertEqual(result.loc[0, "smiles"], "not-a-smiles")
            self.assertEqual(result.loc[0, "molar_mass"], "?")
            self.assertIn("Invalid SMILES", result.loc[0, "error"])
            self.assertEqual(error_path.read_text(encoding="utf-8"), "not-a-smiles\n")
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

    def test_convertor_reports_invalid_smiles_for_xyz(self):
        from groupy.gp_convertor import Convertor

        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = Path(tmpdir) / "invalid.xyz"

            with self.assertLogs("groupy.gp_convertor", level="WARNING") as log_context:
                result = Convertor.smi_to_xyz("not-a-smiles", str(xyz_path))

            self.assertFalse(result)
            self.assertFalse(xyz_path.exists())
            self.assertIn("Failed to parse SMILES", log_context.output[0])

    def test_convert_file_type_surfaces_missing_openbabel_hint(self):
        from groupy.gp_convertor import Convertor

        original_import = builtins.__import__

        def block_openbabel(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "openbabel":
                raise ImportError("blocked for test")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_openbabel
        try:
            with self.assertRaisesRegex(ImportError, "conda install -c conda-forge openbabel"):
                Convertor.convert_file_type("xyz", "missing.xyz", "mol")
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

    def test_batch_smi_to_xyz_can_run_quietly_for_gui_use(self):
        from groupy.gp_convertor import Convertor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            xyz_root = tmp_path / "xyz"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")

            convertor = Convertor()

            def fake_smi_to_xyz(smi, xyz_path=None):
                Path(xyz_path).write_text("fake xyz\n", encoding="utf-8")
                return True

            convertor.smi_to_xyz = fake_smi_to_xyz
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                convertor.batch_smi_to_xyz(str(input_path), str(xyz_root), verbose=False)

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue((xyz_root / "0000.xyz").exists())

    def test_batch_convert_file_type_can_run_quietly_for_gui_use(self):
        from groupy.gp_convertor import Convertor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_root = tmp_path / "input"
            output_root = tmp_path / "output"
            input_root.mkdir()
            (input_root / "molecule.xyz").write_text("fake xyz\n", encoding="utf-8")

            convertor = Convertor()

            def fake_convert_file_type(in_format, in_path, out_format, out_path=None):
                Path(out_path).write_text(f"fake {out_format}\n", encoding="utf-8")

            convertor.convert_file_type = fake_convert_file_type
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                convertor.batch_convert_file_type(
                    "xyz",
                    str(input_root),
                    "mol",
                    str(output_root),
                    verbose=False,
                )

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue((output_root / "molecule.mol").exists())

    def test_batch_file_to_smi_can_run_quietly_for_gui_use(self):
        from groupy.gp_convertor import Convertor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_root = tmp_path / "input"
            output_root = tmp_path / "output"
            input_root.mkdir()
            (input_root / "molecule.mol").write_text("fake mol\n", encoding="utf-8")

            convertor = Convertor()
            convertor.file_to_smi = lambda file_path, format=None: "C1CCCC1"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = convertor.batch_file_to_smi(
                    "mol",
                    str(input_root),
                    str(output_root),
                    verbose=False,
                )

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(result, ["C1CCCC1"])
            self.assertEqual((output_root / "SMILES.txt").read_text(encoding="utf-8"), "C1CCCC1\n")

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
    def test_smi_to_gjf_returns_false_when_xyz_conversion_fails(self):
        from groupy import gp_generator
        from groupy.gp_generator import Generator

        original_convertor = gp_generator.Convertor

        class FailingConvertor:
            def smi_to_xyz(self, smi, xyz_path=None):
                return False

        gp_generator.Convertor = FailingConvertor
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                gjf_path = Path(tmpdir) / "molecule.gjf"

                with self.assertLogs("groupy.gp_generator", level="WARNING") as log_context:
                    result = Generator().smi_to_gjf(
                        "C1CCCC1",
                        gjf_path=str(gjf_path),
                        chk_path=str(Path(tmpdir) / "molecule.chk"),
                        charge_and_multiplicity="0 1",
                    )

                self.assertFalse(result)
                self.assertFalse(gjf_path.exists())
                self.assertIn("Failed to generate gjf", log_context.output[0])
        finally:
            gp_generator.Convertor = original_convertor

    def test_generator_rejects_invalid_smiles_for_charge(self):
        from groupy.exceptions import InvalidSmilesError
        from groupy.gp_generator import Generator

        with self.assertRaises(InvalidSmilesError):
            Generator().calculate_charge("not-a-smiles")

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

    def test_batch_smi_to_gjf_can_run_quietly_for_gui_use(self):
        from groupy.gp_generator import Generator

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "smiles.txt"
            gjf_root = tmp_path / "gjf"
            input_path.write_text("C1CCCC1\n", encoding="utf-8")

            generator = Generator()

            def fake_smi_to_gjf(smi, **kwargs):
                Path(kwargs["gjf_path"]).write_text("fake gjf\n", encoding="utf-8")
                return True

            generator.smi_to_gjf = fake_smi_to_gjf
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                generator.batch_smi_to_gjf(str(input_path), str(gjf_root), verbose=False)

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue((gjf_root / "000000.gjf").exists())

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


class ViewerSmokeTests(unittest.TestCase):
    def test_viewer_module_does_not_import_ase_at_import_time(self):
        command = [
            sys.executable,
            "-c",
            "import sys; import groupy.gp_viewer; print('ase' in sys.modules)",
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

    def test_lazy_ase_import_reports_install_hint_when_missing(self):
        from groupy.gp_viewer import _load_ase_read

        original_import = builtins.__import__

        def block_ase(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ase" or name.startswith("ase."):
                raise ImportError("blocked for test")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_ase
        try:
            with self.assertRaisesRegex(ImportError, "conda install -c conda-forge ase"):
                _load_ase_read()
        finally:
            builtins.__import__ = original_import


class GuiSmokeTests(unittest.TestCase):
    def test_gui_module_does_not_import_pyside6_at_import_time(self):
        command = [
            sys.executable,
            "-c",
            "import sys; import groupy.gui; print('PySide6' in sys.modules)",
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

    def test_gui_check_reports_dependency_status(self):
        command = [
            sys.executable,
            "-m",
            "groupy.gui",
            "--check",
        ]

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        if importlib.util.find_spec("PySide6"):
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PySide6 is available.", completed.stdout)
        else:
            self.assertEqual(completed.returncode, 1)
            self.assertIn("python -m pip install -e", completed.stderr)
            self.assertIn(".[gui]", completed.stderr)

    def test_gui_record_helpers_use_public_api(self):
        from groupy.gui.app import calculate_records, count_records

        calculation = calculate_records(["C1CCCC1"])
        counts = count_records(["C1CCCC1"])

        self.assertEqual(calculation[0]["smiles"], "C1CCCC1")
        self.assertAlmostEqual(calculation[0]["molar_mass"], 70.135)
        self.assertEqual(counts[0], {"f_168": 5, "smiles": "C1CCCC1"})


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

    def test_calculate_cli_reports_invalid_smiles_as_json(self):
        command = [
            sys.executable,
            "-m",
            "groupy.cli",
            "calculate",
            "--smiles",
            "not-a-smiles",
        ]

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        result = json.loads(completed.stdout)
        self.assertEqual(result["smiles"], "not-a-smiles")
        self.assertEqual(result["molar_mass"], "?")
        self.assertIn("Invalid SMILES", result["error"])

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
