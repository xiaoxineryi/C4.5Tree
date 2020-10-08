import sys
from PyQt5.QtWidgets import *

class TreeWidget(QTreeWidget):
    def __init__(self,data):
        super(TreeWidget, self).__init__()

        self.setColumnCount(2)  # 共2列
        self.setHeaderLabels(['Key', 'Value'])

        self.rootList = []
        root = self
        self.generateTreeWidget(data, root)

        print(len(self.rootList), self.rootList)

        self.insertTopLevelItems(0, self.rootList)

    def generateTreeWidget(self, data, root):
        if isinstance(data, dict):
            for key in data.keys():
                child = QTreeWidgetItem()
                child.setText(0, str(key))
                if isinstance(root, QTreeWidget) == False:  # 非根节点，添加子节点
                    root.addChild(child)
                self.rootList.append(child)
                print(key)
                value = data[key]
                self.generateTreeWidget(value, child)
        else:
            root.setText(1, str(data))


def printTree(data):
    app = QApplication(sys.argv)
    win = TreeWidget(data)
    win.show()
    sys.exit(app.exec_())