import sys
from PySide6.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("파이참 테스트 창")
window.resize(500, 400)
window.show()
sys.exit(app.exec())


# import sys
# from PySide6.QtWidgets import QApplication
# from PySide6.QtUiTools import QUiLoader
# from PySide6.QtCore import QFile
# from qt_material import apply_stylesheet
#
#
# def main():
#     app = QApplication(sys.argv)
#
#     # 🌟 Qt-Material 테마 적용 (다크 청록색)
#     apply_stylesheet(app, theme='dark_teal.xml')
#
#     loader = QUiLoader()
#     # 경로를 Raw String(r"...")으로 지정
#     ui_file_path = r"C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\hydro_prg_2025_v12_add_3btn_ai_20251219_v12.ui"
#     path = QFile(ui_file_path)
#
#     # 1. 파일 열기 시도 및 예외 처리 (한 번만 실행!)
#     if not path.open(QFile.ReadOnly):
#         print(f"🚨 UI 파일을 찾을 수 없거나 열 수 없습니다.\n경로를 확인하세요: {ui_file_path}")
#         sys.exit(-1)
#
#     # 2. UI 파일 전체를 불러와서 window 변수에 통째로 담기
#     window = loader.load(path)
#     path.close()
#
#     if not window:
#         print("🚨 UI 파일을 성공적으로 읽었지만, 화면 객체를 생성하지 못했습니다.")
#         sys.exit(-1)
#
#     # 3. 화면 띄우기
#     window.show()
#     sys.exit(app.exec())
#
#
# if __name__ == '__main__':
#     main()