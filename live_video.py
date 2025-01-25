import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog


class LiveVideo:
    def __init__(self, window):
        self.window = window
        self.window.title('Live Video')

        self.cap = cv2.VideoCapture('datasets/full3.mp4')

        self.video_label = ttk.Label(self.window)
        self.video_label.grid(row=0, column=0, padx=5, pady=5)

        self.video_entry = tk.Entry(root, width=50)
        self.video_entry.grid(row=1, column=0, padx=5, pady=5)

        self.browse_button = tk.Button(root, text="Brose", command=self.browse_video)
        self.browse_button.grid(row=1, column=1, padx=5, pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=1, column=2, padx=5,pady=5)


    def browse_video(self):
        filename = filedialog.askopenfilename(title='Select Video File', filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))

        if filename:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, filename)
            self.start_processing()

    def start_processing(self):
        video_path = self.video_entry.get()
        if not video_path:
            return

        if not os.path.isfile(video_path):
            print('Invalid file path')
            return

        self.cap = cv2.VideoCapture(video_path)
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        self.window.after(10, self.update)
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #
        #     cv2.imshow('frame', frame)
        #
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        #
        # cap.release()
        # cv2.destroyAllWindows()

    def display_frame(self, frame):
        img = cv2.resize(frame, (640, 480))
        imgtk = cv2.imencode('.png', img)[1].tobytes()
        tkimg = tk.PhotoImage(data=imgtk)
        self.video_label.imgtk = tkimg
        self.video_label.configure(image=tkimg)

    def quit_app(self):
        self.cap.release()
        self.window.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = LiveVideo(root)
    root.mainloop()