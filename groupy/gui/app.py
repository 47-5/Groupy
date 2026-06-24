"""Minimal PySide6 desktop application for Groupy."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from groupy.api import calculate_smiles, count_smiles
from groupy.io import write_records_csv

INSTALL_HINT = (
    "PySide6 is required for the Groupy desktop GUI. "
    "Install it with `python -m pip install -e \".[gui]\"`."
)


def is_pyside6_available() -> bool:
    """Return whether PySide6 can be imported in the current environment."""
    return importlib.util.find_spec("PySide6") is not None


def calculate_records(smiles_values: list[str]) -> list[dict[str, Any]]:
    """Calculate properties for a list of SMILES strings."""
    return [calculate_smiles(smiles) for smiles in smiles_values]


def count_records(smiles_values: list[str]) -> list[dict[str, Any]]:
    """Count groups for a list of SMILES strings."""
    return [count_smiles(smiles) for smiles in smiles_values]


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

    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = _create_main_window(QtCore, QtWidgets)
    window.show()
    return app.exec()


def _create_main_window(QtCore: Any, QtWidgets: Any):
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
            self._worker = None
            self._worker_thread = None
            self.setWindowTitle("Groupy")
            self.resize(980, 680)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root_layout = QtWidgets.QVBoxLayout(central)
            root_layout.setContentsMargins(14, 14, 14, 14)
            root_layout.setSpacing(10)

            input_label = QtWidgets.QLabel("SMILES")
            root_layout.addWidget(input_label)

            self.smiles_input = QtWidgets.QPlainTextEdit()
            self.smiles_input.setPlaceholderText("Enter one SMILES per line")
            self.smiles_input.setPlainText("C1CCCC1")
            self.smiles_input.setMinimumHeight(96)
            root_layout.addWidget(self.smiles_input)

            action_layout = QtWidgets.QHBoxLayout()
            root_layout.addLayout(action_layout)

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

            self.result_table = QtWidgets.QTableWidget()
            self.result_table.setAlternatingRowColors(True)
            self.result_table.setSortingEnabled(True)
            self.result_table.horizontalHeader().setStretchLastSection(True)
            self.result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            root_layout.addWidget(self.result_table, stretch=1)

            self.statusBar().showMessage("Ready")

        def _smiles_values(self) -> list[str]:
            values = [line.strip() for line in self.smiles_input.toPlainText().splitlines() if line.strip()]
            if not values:
                raise ValueError("Enter at least one SMILES string.")
            return values

        def _calculate_properties(self) -> None:
            self._start_worker(calculate_records, "Calculated properties", "Calculating properties...")

        def _count_groups(self) -> None:
            self._start_worker(count_records, "Counted groups", "Counting groups...")

        def _start_worker(self, operation, done_message: str, progress_message: str) -> None:
            if self._worker_thread is not None:
                return
            try:
                smiles_values = self._smiles_values()
            except Exception as exc:
                self._show_error(str(exc))
                return

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
            self.calculate_button.setEnabled(not busy)
            self.count_button.setEnabled(not busy)
            self.smiles_input.setEnabled(not busy)
            self.export_button.setEnabled((not busy) and bool(self._records))
            if busy:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.statusBar().showMessage(message or "Working...")
            else:
                QtWidgets.QApplication.restoreOverrideCursor()

        def _show_records(self, records: list[dict[str, Any]], message: str) -> None:
            self._records = records
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
            self.statusBar().showMessage(f"{message}: {len(records)} molecule(s)")

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
