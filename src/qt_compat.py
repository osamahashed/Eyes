"""Qt compatibility layer between PyQt6 and PySide6."""

try:
    from PyQt6 import QtWidgets, QtGui, QtCore
    from PyQt6.QtCore import Qt
except Exception:
    try:
        from PySide6 import QtWidgets, QtGui, QtCore
        from PySide6.QtCore import Qt
    except Exception as e:
        raise ImportError(
            "Neither PyQt6 nor PySide6 could be imported. "
            "Install one of them (e.g., pip install PyQt6 or pip install PySide6)."
        )

# Normalize differences
QtWidgets.QApplication = getattr(QtWidgets, 'QApplication', QtWidgets.QApplication)
QtWidgets.QMainWindow = getattr(QtWidgets, 'QMainWindow', QtWidgets.QMainWindow)
QtWidgets.QLabel = getattr(QtWidgets, 'QLabel', QtWidgets.QLabel)
QtWidgets.QVBoxLayout = getattr(QtWidgets, 'QVBoxLayout', QtWidgets.QVBoxLayout)
QtWidgets.QHBoxLayout = getattr(QtWidgets, 'QHBoxLayout', QtWidgets.QHBoxLayout)
QtWidgets.QGridLayout = getattr(QtWidgets, 'QGridLayout', QtWidgets.QGridLayout)
QtWidgets.QPushButton = getattr(QtWidgets, 'QPushButton', QtWidgets.QPushButton)
QtWidgets.QProgressBar = getattr(QtWidgets, 'QProgressBar', QtWidgets.QProgressBar)
QtWidgets.QFrame = getattr(QtWidgets, 'QFrame', QtWidgets.QFrame)
QtWidgets.QGroupBox = getattr(QtWidgets, 'QGroupBox', QtWidgets.QGroupBox)
QtWidgets.QSpacerItem = getattr(QtWidgets, 'QSpacerItem', QtWidgets.QSpacerItem)
QtWidgets.QSizePolicy = getattr(QtWidgets, 'QSizePolicy', QtWidgets.QSizePolicy)
QtWidgets.QWidget = getattr(QtWidgets, 'QWidget', QtWidgets.QWidget)
QtWidgets.QSlider = getattr(QtWidgets, 'QSlider', QtWidgets.QSlider)
QtWidgets.QCheckBox = getattr(QtWidgets, 'QCheckBox', QtWidgets.QCheckBox)
QtWidgets.QComboBox = getattr(QtWidgets, 'QComboBox', QtWidgets.QComboBox)
QtCore.QTimer = getattr(QtCore, 'QTimer', QtCore.QTimer)

QtGui.QImage = getattr(QtGui, 'QImage', QtGui.QImage)
QtGui.QPixmap = getattr(QtGui, 'QPixmap', QtGui.QPixmap)
QtGui.QPainter = getattr(QtGui, 'QPainter', QtGui.QPainter)
QtGui.QPen = getattr(QtGui, 'QPen', QtGui.QPen)
QtGui.QColor = getattr(QtGui, 'QColor', QtGui.QColor)
QtGui.QBrush = getattr(QtGui, 'QBrush', QtGui.QBrush)
QtGui.QFont = getattr(QtGui, 'QFont', QtGui.QFont)

QtCore.Qt = getattr(QtCore, 'Qt', QtCore.Qt)

# Export common aliases so callers can import either the namespaces or widgets directly.
QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QLabel = QtWidgets.QLabel
QVBoxLayout = QtWidgets.QVBoxLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QGridLayout = QtWidgets.QGridLayout
QPushButton = QtWidgets.QPushButton
QProgressBar = QtWidgets.QProgressBar
QFrame = QtWidgets.QFrame
QGroupBox = QtWidgets.QGroupBox
QSpacerItem = QtWidgets.QSpacerItem
QSizePolicy = QtWidgets.QSizePolicy
QWidget = QtWidgets.QWidget
QSlider = QtWidgets.QSlider
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QTimer = QtCore.QTimer
QImage = QtGui.QImage
QPixmap = QtGui.QPixmap
QPainter = QtGui.QPainter
QPen = QtGui.QPen
QColor = QtGui.QColor
QBrush = QtGui.QBrush
QFont = QtGui.QFont
