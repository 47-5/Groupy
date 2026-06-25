"""Minimal PySide6 desktop application for Groupy."""

from __future__ import annotations

import argparse
from io import BytesIO
import importlib.util
import sys
from pathlib import Path
from typing import Any

from groupy.chem import ensure_mol
from groupy.api import calculate_smiles, count_smiles
from groupy.io import load_smiles_file, write_records_csv

INSTALL_HINT = (
    "PySide6 is required for the Groupy desktop GUI. "
    "Install it with `python -m pip install -e \".[gui]\"`."
)


def is_pyside6_available() -> bool:
    """Return whether PySide6 can be imported in the current environment."""
    return importlib.util.find_spec("PySide6") is not None


def calculate_records(
    smiles_values: list[str],
    *,
    parameter_type: str = "step_wise",
    check_hydrocarbon: bool = True,
) -> list[dict[str, Any]]:
    """Calculate properties for a list of SMILES strings."""
    return [
        calculate_smiles(
            smiles,
            parameter_type=parameter_type,
            check_hydrocarbon=check_hydrocarbon,
        )
        for smiles in smiles_values
    ]


def count_records(
    smiles_values: list[str],
    *,
    include_zero: bool = False,
    include_smiles: bool = True,
) -> list[dict[str, Any]]:
    """Count groups for a list of SMILES strings."""
    return [
        count_smiles(
            smiles,
            include_zero=include_zero,
            include_smiles=include_smiles,
        )
        for smiles in smiles_values
    ]


def load_smiles_text(path: str | Path) -> str:
    """Load a SMILES file and return text suitable for the GUI editor."""
    return "\n".join(load_smiles_file(path))


def render_smiles_png(smiles: str, *, width: int = 360, height: int = 280) -> bytes:
    """Render one SMILES string to PNG bytes using RDKit."""
    from rdkit.Chem import Draw

    molecule = ensure_mol(smiles)
    image = Draw.MolToImage(molecule, size=(width, height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    """Run the desktop GUI."""
    parser = argparse.ArgumentParser(prog="Groupy-GUI", description="Launch the Groupy desktop GUI.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether GUI dependencies are installed without launching a window.",
    )
    args = parser.parse_args(argv)

    if not is_pyside6_available():
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    if args.check:
        print("PySide6 is available.")
        return 0

    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = _create_main_window(QtCore, QtGui, QtWidgets)
    window.show()
    return app.exec()


def _create_main_window(QtCore: Any, QtGui: Any, QtWidgets: Any):
    class BatchWorker(QtCore.QObject):
        finished = QtCore.Signal(object, str)
        failed = QtCore.Signal(str)

        def __init__(self, smiles_values, operation, message):
            super().__init__()
            self._smiles_values = smiles_values
            self._operation = operation
            self._message = message

        @QtCore.Slot()
        def run(self):
            try:
                records = self._operation(self._smiles_values)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            self.finished.emit(records, self._message)

    class GroupyMainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self._records: list[dict[str, Any]] = []
            self._record_smiles_values: list[str] = []
            self._pending_smiles_values: list[str] = []
            self._structure_pixmap = None
            self._worker = None
            self._worker_thread = None
            self.setWindowTitle("Groupy")
            self.resize(1180, 720)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root_layout = QtWidgets.QVBoxLayout(central)
            root_layout.setContentsMargins(14, 14, 14, 14)
            root_layout.setSpacing(10)

            content_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            root_layout.addWidget(content_splitter, stretch=1)

            work_panel = QtWidgets.QWidget()
            work_layout = QtWidgets.QVBoxLayout(work_panel)
            work_layout.setContentsMargins(0, 0, 10, 0)
            work_layout.setSpacing(10)
            content_splitter.addWidget(work_panel)

            preview_panel = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_panel)
            preview_layout.setContentsMargins(10, 0, 0, 0)
            preview_layout.setSpacing(8)
            content_splitter.addWidget(preview_panel)

            input_label = QtWidgets.QLabel("SMILES")
            work_layout.addWidget(input_label)

            self.smiles_input = QtWidgets.QPlainTextEdit()
            self.smiles_input.setPlaceholderText("Enter one SMILES per line")
            self.smiles_input.setPlainText("C1CCCC1")
            self.smiles_input.setMinimumHeight(96)
            self.smiles_input.textChanged.connect(self._preview_first_input_smiles)
            work_layout.addWidget(self.smiles_input)

            action_layout = QtWidgets.QHBoxLayout()
            work_layout.addLayout(action_layout)

            self.import_button = QtWidgets.QPushButton("Import File")
            self.import_button.clicked.connect(self._import_smiles_file)
            action_layout.addWidget(self.import_button)

            self.calculate_button = QtWidgets.QPushButton("Calculate Properties")
            self.calculate_button.clicked.connect(self._calculate_properties)
            action_layout.addWidget(self.calculate_button)

            self.count_button = QtWidgets.QPushButton("Count Groups")
            self.count_button.clicked.connect(self._count_groups)
            action_layout.addWidget(self.count_button)

            self.export_button = QtWidgets.QPushButton("Export CSV")
            self.export_button.setEnabled(False)
            self.export_button.clicked.connect(self._export_csv)
            action_layout.addWidget(self.export_button)

            action_layout.addStretch(1)

            options_layout = QtWidgets.QHBoxLayout()
            work_layout.addLayout(options_layout)

            options_layout.addWidget(QtWidgets.QLabel("Parameters"))
            self.parameter_type_combo = QtWidgets.QComboBox()
            self.parameter_type_combo.addItem("Stepwise", "step_wise")
            self.parameter_type_combo.addItem("Simultaneous", "simultaneous")
            options_layout.addWidget(self.parameter_type_combo)

            self.check_hydrocarbon_box = QtWidgets.QCheckBox("Hydrocarbon check")
            self.check_hydrocarbon_box.setChecked(True)
            options_layout.addWidget(self.check_hydrocarbon_box)

            self.include_zero_box = QtWidgets.QCheckBox("Show zero groups")
            options_layout.addWidget(self.include_zero_box)

            self.include_smiles_box = QtWidgets.QCheckBox("Include SMILES")
            self.include_smiles_box.setChecked(True)
            options_layout.addWidget(self.include_smiles_box)

            options_layout.addStretch(1)

            self.result_table = QtWidgets.QTableWidget()
            self.result_table.setAlternatingRowColors(True)
            self.result_table.setSortingEnabled(True)
            self.result_table.horizontalHeader().setStretchLastSection(True)
            self.result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.result_table.itemSelectionChanged.connect(self._preview_selected_record)
            work_layout.addWidget(self.result_table, stretch=1)

            preview_label = QtWidgets.QLabel("Structure")
            preview_layout.addWidget(preview_label)

            self.structure_view = QtWidgets.QLabel()
            self.structure_view.setAlignment(QtCore.Qt.AlignCenter)
            self.structure_view.setMinimumSize(320, 260)
            self.structure_view.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.structure_view.setText("No structure")
            preview_layout.addWidget(self.structure_view, stretch=1)

            preview_layout.addStretch(1)
            content_splitter.setStretchFactor(0, 4)
            content_splitter.setStretchFactor(1, 2)

            self._busy_controls = [
                self.import_button,
                self.calculate_button,
                self.count_button,
                self.smiles_input,
                self.parameter_type_combo,
                self.check_hydrocarbon_box,
                self.include_zero_box,
                self.include_smiles_box,
            ]
            self._preview_first_input_smiles()

            self.statusBar().showMessage("Ready")

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._refresh_structure_pixmap()

        def _smiles_values(self) -> list[str]:
            values = [line.strip() for line in self.smiles_input.toPlainText().splitlines() if line.strip()]
            if not values:
                raise ValueError("Enter at least one SMILES string.")
            return values

        def _calculate_properties(self) -> None:
            parameter_type = self.parameter_type_combo.currentData()
            check_hydrocarbon = self.check_hydrocarbon_box.isChecked()
            self._start_worker(
                lambda smiles_values: calculate_records(
                    smiles_values,
                    parameter_type=parameter_type,
                    check_hydrocarbon=check_hydrocarbon,
                ),
                "Calculated properties",
                "Calculating properties...",
            )

        def _count_groups(self) -> None:
            include_zero = self.include_zero_box.isChecked()
            include_smiles = self.include_smiles_box.isChecked()
            self._start_worker(
                lambda smiles_values: count_records(
                    smiles_values,
                    include_zero=include_zero,
                    include_smiles=include_smiles,
                ),
                "Counted groups",
                "Counting groups...",
            )

        def _start_worker(self, operation, done_message: str, progress_message: str) -> None:
            if self._worker_thread is not None:
                return
            try:
                smiles_values = self._smiles_values()
            except Exception as exc:
                self._show_error(str(exc))
                return

            self._pending_smiles_values = smiles_values
            self._set_busy(True, progress_message)
            thread = QtCore.QThread(self)
            worker = BatchWorker(smiles_values, operation, done_message)
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(self._finish_worker)
            worker.failed.connect(self._fail_worker)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            worker.finished.connect(thread.quit)
            worker.failed.connect(thread.quit)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_worker)

            self._worker = worker
            self._worker_thread = thread
            thread.start()

        def _finish_worker(self, records: list[dict[str, Any]], message: str) -> None:
            self._set_busy(False)
            self._show_records(records, message)

        def _fail_worker(self, message: str) -> None:
            self._set_busy(False)
            self._show_error(message)

        def _clear_worker(self) -> None:
            self._worker = None
            self._worker_thread = None

        def _set_busy(self, busy: bool, message: str | None = None) -> None:
            for control in self._busy_controls:
                control.setEnabled(not busy)
            self.export_button.setEnabled((not busy) and bool(self._records))
            if busy:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.statusBar().showMessage(message or "Working...")
            else:
                QtWidgets.QApplication.restoreOverrideCursor()

        def _show_records(self, records: list[dict[str, Any]], message: str) -> None:
            self._records = records
            self._record_smiles_values = list(self._pending_smiles_values)
            columns = _record_columns(records)
            self.result_table.setSortingEnabled(False)
            self.result_table.clear()
            self.result_table.setRowCount(len(records))
            self.result_table.setColumnCount(len(columns))
            self.result_table.setHorizontalHeaderLabels(columns)

            for row_index, record in enumerate(records):
                for column_index, column in enumerate(columns):
                    value = record.get(column, "")
                    item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.result_table.setItem(row_index, column_index, item)

            self.result_table.resizeColumnsToContents()
            self.result_table.setSortingEnabled(True)
            self.export_button.setEnabled(bool(records))
            if records:
                self.result_table.selectRow(0)
                self._preview_record(0)
            else:
                self._preview_first_input_smiles()
            self.statusBar().showMessage(f"{message}: {len(records)} molecule(s)")

        def _import_smiles_file(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Import SMILES",
                str(Path.cwd()),
                "SMILES files (*.txt *.csv *.xlsx);;All files (*.*)",
            )
            if not path:
                return
            try:
                text = load_smiles_text(path)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self.smiles_input.setPlainText(text)
            count = len([line for line in text.splitlines() if line.strip()])
            self.statusBar().showMessage(f"Imported {count} molecule(s) from {path}")

        def _export_csv(self) -> None:
            if not self._records:
                self._show_error("No results to export.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export CSV",
                str(Path.cwd() / "groupy_results.csv"),
                "CSV files (*.csv)",
            )
            if not path:
                return
            write_records_csv(self._records, path)
            self.statusBar().showMessage(f"Exported {path}")

        def _show_error(self, message: str) -> None:
            QtWidgets.QMessageBox.warning(self, "Groupy", message)
            self.statusBar().showMessage(message)

        def _preview_first_input_smiles(self) -> None:
            values = [line.strip() for line in self.smiles_input.toPlainText().splitlines() if line.strip()]
            if not values:
                self._clear_structure("No structure")
                return
            self._show_structure(values[0])

        def _preview_selected_record(self) -> None:
            selected_rows = self.result_table.selectionModel().selectedRows()
            if not selected_rows:
                return
            self._preview_record(selected_rows[0].row())

        def _preview_record(self, row: int) -> None:
            if row < 0 or row >= len(self._records):
                return
            smiles = _record_smiles(self._records[row])
            if smiles is None and row < len(self._record_smiles_values):
                smiles = self._record_smiles_values[row]
            if not smiles:
                self._clear_structure("No SMILES")
                return
            self._show_structure(smiles)

        def _show_structure(self, smiles: str) -> None:
            try:
                image_bytes = render_smiles_png(smiles)
            except Exception:
                self._clear_structure("Invalid SMILES")
                return
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(image_bytes, "PNG")
            self._structure_pixmap = pixmap
            self._refresh_structure_pixmap()

        def _refresh_structure_pixmap(self) -> None:
            if self._structure_pixmap is None:
                return
            scaled = self._structure_pixmap.scaled(
                self.structure_view.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.structure_view.setPixmap(scaled)

        def _clear_structure(self, message: str) -> None:
            self._structure_pixmap = None
            self.structure_view.clear()
            self.structure_view.setText(message)

    return GroupyMainWindow()


def _record_columns(records: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for preferred in ("smiles", "note", "error"):
        if any(preferred in record for record in records):
            columns.append(preferred)
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    return columns


def _record_smiles(record: dict[str, Any]) -> str | None:
    for key in ("smiles", "note"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value.split()[0]
    return None
