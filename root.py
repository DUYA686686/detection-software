from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices

path = r"./"
QDesktopServices.openUrl(QUrl.fromLocalFile(path))