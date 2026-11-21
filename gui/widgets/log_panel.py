#!/usr/bin/env python3
"""Scrollable status/log panel with a progress bar."""

from ..qt_compat import QtCore, QtGui, QtWidgets, Qt

LEVEL_COLORS = {
    "info": "#c8d3e6",
    "warn": "#ffcc66",
    "error": "#ff6b6b",
    "ok": "#6ee7a0",
    "cmd": "#7fd1ff",
}
MAX_BLOCKS = 5000          # keep the document bounded on a long batch


class LogPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Status log")
        title.setStyleSheet("font-weight:600;")
        header.addWidget(title)
        header.addStretch(1)

        self.autoscroll = QtWidgets.QCheckBox("Auto-scroll")
        self.autoscroll.setChecked(True)
        header.addWidget(self.autoscroll)

        clear = QtWidgets.QToolButton()
        clear.setText("Clear")
        clear.clicked.connect(self.clear)
        header.addWidget(clear)

        copy = QtWidgets.QToolButton()
        copy.setText("Copy")
        copy.clicked.connect(self._copy_all)
        header.addWidget(copy)
        layout.addLayout(header)

        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(MAX_BLOCKS)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(9)
        self.view.setFont(font)
        self.view.setStyleSheet(
            "QPlainTextEdit{background:#12161f;color:#c8d3e6;"
            "border:1px solid #2a3347;border-radius:4px;}")
        layout.addWidget(self.view, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("idle")
        layout.addWidget(self.progress)

    # -- API ---------------------------------------------------------------- #
    def append(self, level, text):
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
        stamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        safe = (str(text).replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
        self.view.appendHtml(
            '<span style="color:#5d6b85;">%s</span> '
            '<span style="color:%s;">%s</span>' % (stamp, color, safe))
        if self.autoscroll.isChecked():
            bar = self.view.verticalScrollBar()
            bar.setValue(bar.maximum())

    def info(self, text):
        self.append("info", text)

    def ok(self, text):
        self.append("ok", text)

    def warn(self, text):
        self.append("warn", text)

    def error(self, text):
        self.append("error", text)

    def command(self, text):
        self.append("cmd", text)

    def set_progress(self, pct, message=""):
        self.progress.setValue(max(0, min(100, int(pct))))
        self.progress.setFormat(message or "%p%")

    def reset_progress(self, text="idle"):
        self.progress.setValue(0)
        self.progress.setFormat(text)

    def clear(self):
        self.view.clear()

    def _copy_all(self):
        QtWidgets.QApplication.clipboard().setText(self.view.toPlainText())
        self.info("Log copied to clipboard.")
