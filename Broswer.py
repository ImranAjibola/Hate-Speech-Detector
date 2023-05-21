import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar, QAction, QLineEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView

class BrowserWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Web Browser')
        self.setGeometry(100, 100, 800, 600)

        self.web_view = QWebEngineView()
        self.setCentralWidget(self.web_view)

        self.toolbar = QToolBar('Toolbar')
        self.addToolBar(self.toolbar)

        self.back_action = QAction('Back', self)
        self.back_action.triggered.connect(self.web_view.back)
        self.toolbar.addAction(self.back_action)

        self.forward_action = QAction('Forward', self)
        self.forward_action.triggered.connect(self.web_view.forward)
        self.toolbar.addAction(self.forward_action)

        self.refresh_action = QAction('Refresh', self)
        self.refresh_action.triggered.connect(self.web_view.reload)
        self.toolbar.addAction(self.refresh_action)

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.load_url)
        self.toolbar.addWidget(self.url_bar)

    def load_url(self):
        query = self.url_bar.text()

        if query.startswith('http'):
            self.web_view.load(QUrl(query))
        else:
            search_url = 'https://www.google.com/search?q=' + query
            self.web_view.load(QUrl(search_url))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BrowserWindow()
    window.show()

    window.load_url()  # Load the initial URL or perform a search, e.g., window.load_url('https://www.google.com') or window.load_url('OpenAI')

    sys.exit(app.exec_())
# import sys
# import pickle
# from PyQt5.QtCore import QUrl
# from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar, QAction, QLineEdit
# from PyQt5.QtWebEngineWidgets import QWebEngineView
# from bs4 import BeautifulSoup

# class BrowserWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle('Web Browser')
#         self.setGeometry(100, 100, 800, 600)

#         self.web_view = QWebEngineView()
#         self.setCentralWidget(self.web_view)

#         self.toolbar = QToolBar('Toolbar')
#         self.addToolBar(self.toolbar)

#         self.back_action = QAction('Back', self)
#         self.back_action.triggered.connect(self.web_view.back)
#         self.toolbar.addAction(self.back_action)

#         self.forward_action = QAction('Forward', self)
#         self.forward_action.triggered.connect(self.web_view.forward)
#         self.toolbar.addAction(self.forward_action)

#         self.refresh_action = QAction('Refresh', self)
#         self.refresh_action.triggered.connect(self.web_view.reload)
#         self.toolbar.addAction(self.refresh_action)

#         self.url_bar = QLineEdit()
#         self.url_bar.returnPressed.connect(self.load_url)
#         self.toolbar.addWidget(self.url_bar)

#         # Load the Hate Speech Detection Model from the .pkl file
#         with open(r'C:\Users\Imran Ajibola\Desktop\RAIN\2nd semester\project\Hate Speech Detection\Hate_speech.pkl', 'rb') as f:
#             self.model = pickle.load(f)

#     def load_url(self):
#         query = self.url_bar.text()

#         if query.startswith('http'):
#             self.web_view.load(QUrl(query))
#         else:
#             search_url = 'https://www.google.com/search?q=' + query
#             self.web_view.load(QUrl(search_url))

#         self.web_view.loadFinished.connect(self.process_web_page)

#     def process_web_page(self):
#         page = self.web_view.page()
#         page.toHtml(self.handle_html)

#     def handle_html(self, html):
#         soup = BeautifulSoup(html, 'html.parser')
#         sentences = soup.get_text().split('.')  # Split the text into sentences (naive approach)

#         for i, sentence in enumerate(sentences):
#             # Use your hate speech detection model to predict if the sentence contains hate speech
#             is_hate_speech = self.model.predict([sentence])[0]

#             if is_hate_speech:
#                 # Perform the action to block or blur the hate speech sentence
#                 sentences[i] = '***HATE SPEECH***'

#         modified_html = ' '.join(sentences)
#         self.web_view.setHtml(modified_html)

# if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # window = BrowserWindow()
    # window.show()

    # window.load_url()  # Load the initial URL or perform a search

    # sys.exit(app.exec_())
