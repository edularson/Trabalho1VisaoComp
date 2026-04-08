import cv2
import numpy as np
import time
import pyautogui
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import os


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class VisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Visão Computacional - Dashboard Pro")
        self.geometry("1200x900")

        # Variáveis de Controle
        self.path_img1 = ""
        self.path_img2 = ""
        self.cap = None
        self.is_camera_on = False
        self.last_action = time.time()
        
        # Variáveis para Optical Flow
        self.old_gray = None
        self.p0 = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # layout principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="CONTROLES", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        # secao panorama
        self.btn_load1 = ctk.CTkButton(self.sidebar, text="1. Imagem Esquerda", command=lambda: self.load_file(1), fg_color="#3b3b3b")
        self.btn_load1.pack(pady=5, padx=10)
        self.btn_load2 = ctk.CTkButton(self.sidebar, text="2. Imagem Direita", command=lambda: self.load_file(2), fg_color="#3b3b3b")
        self.btn_load2.pack(pady=5, padx=10)

        self.feat_choice = ctk.CTkOptionMenu(self.sidebar, values=["SIFT", "ORB"])
        self.feat_choice.pack(pady=10)
        self.match_choice = ctk.CTkOptionMenu(self.sidebar, values=["BF", "FLANN"])
        self.match_choice.pack(pady=5)

        self.btn_run_pano = ctk.CTkButton(self.sidebar, text="GERAR PANORAMA", command=self.run_stitching, fg_color="green")
        self.btn_run_pano.pack(pady=20, padx=10)

        # secao gestos
        ctk.CTkLabel(self.sidebar, text="------------------").pack()
        self.btn_toggle_cam = ctk.CTkButton(self.sidebar, text="LIGAR WEBCAM", command=self.toggle_camera, fg_color="#1f538d")
        self.btn_toggle_cam.pack(pady=20, padx=10)

        #mensagem gesto
        self.label_gesture_msg = ctk.CTkLabel(self.sidebar, text="", font=ctk.CTkFont(size=16, weight="bold"))
        self.label_gesture_msg.pack(pady=10)

        self.main_view = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # webcam
        ctk.CTkLabel(self.main_view, text="FEED EM TEMPO REAL (GESTOS)", font=ctk.CTkFont(weight="bold")).pack()
        self.canvas_cam = ctk.CTkLabel(self.main_view, text="Câmera Desligada", fg_color="#1a1a1a", width=640, height=360, corner_radius=10)
        self.canvas_cam.pack(pady=10)

        # resultados panoram
        ctk.CTkLabel(self.main_view, text="RESULTADOS DO PANORAMA", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 0))
        self.canvas_pre = ctk.CTkLabel(self.main_view, text="Matches (Pré)", fg_color="#1a1a1a", width=700, height=300, corner_radius=10)
        self.canvas_pre.pack(pady=5)
        self.canvas_pos = ctk.CTkLabel(self.main_view, text="Panorama (Pós)", fg_color="#1a1a1a", width=700, height=300, corner_radius=10)
        self.canvas_pos.pack(pady=5)

    def toggle_camera(self):
        if not self.is_camera_on:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(1)
            
            if self.cap.isOpened():
                self.is_camera_on = True
                self.btn_toggle_cam.configure(text="DESLIGAR WEBCAM", fg_color="red")
                self.update_camera_feed()
            else:
                messagebox.showerror("Erro", "Câmera não encontrada.")
        else:
            self.is_camera_on = False
            self.btn_toggle_cam.configure(text="LIGAR WEBCAM", fg_color="#1f538d")
            if self.cap: self.cap.release()
            self.canvas_cam.configure(image="", text="Câmera Desligada")

    def show_gesture_feedback(self, text, color):
        self.label_gesture_msg.configure(text=text, text_color=color)
        self.after(1000, lambda: self.label_gesture_msg.configure(text=""))

    def update_camera_feed(self):
        if self.is_camera_on and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.old_gray is None:
                    self.old_gray = gray.copy()
                    self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)

                if self.p0 is not None:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **self.lk_params)
                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = self.p0[st == 1]

                        if len(good_new) > 10:
                            dx = np.mean(good_new[:, 0] - good_old[:, 0])
                            
                            if time.time() - self.last_action > 1.3:
                                if dx > 20:
                                    pyautogui.press('right')
                                    self.show_gesture_feedback(">> AVANÇAR", "green")
                                    self.last_action = time.time()
                                elif dx < -20:
                                    pyautogui.press('left')
                                    self.show_gesture_feedback("<< VOLTAR", "orange")
                                    self.last_action = time.time()

                            for pt in good_new:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

                        if len(good_new) > 20:
                            self.p0 = good_new.reshape(-1, 1, 2)
                        else:
                            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)

                self.old_gray = gray.copy()

                img_tk = self.cv2_to_ctk(frame, 640, 360)
                self.canvas_cam.configure(image=img_tk, text="")

            self.after(10, self.update_camera_feed)

    def cv2_to_ctk(self, cv_img, width, height):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(width, height))

    def load_file(self, num):
        path = filedialog.askopenfilename()
        if path:
            if num == 1: self.path_img1 = path
            else: self.path_img2 = path
            messagebox.showinfo("Sucesso", f"Imagem {num} carregada!")

    def run_stitching(self):
        if not self.path_img1 or not self.path_img2:
            messagebox.showwarning("Aviso", "Selecione as fotos!")
            return
        
        img_l = cv2.imread(self.path_img1)
        img_r = cv2.imread(self.path_img2)
        feat, match = self.feat_choice.get(), self.match_choice.get()

        detector = cv2.SIFT_create() if feat == "SIFT" else cv2.ORB_create(nfeatures=2000)
        kp1, des1 = detector.detectAndCompute(img_l, None)
        kp2, des2 = detector.detectAndCompute(img_r, None)

        if match == 'BF':
            matcher = cv2.BFMatcher(cv2.NORM_L2 if feat == 'SIFT' else cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
        else:
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1 if feat=='SIFT' else 6, trees=5), dict(checks=50))
            matches = [m for m, n in matcher.knnMatch(des1, des2, k=2) if m.distance < 0.7 * n.distance]

        if len(matches) > 4:
            img_m = cv2.drawMatches(img_l, kp1, img_r, kp2, matches[:50], None, flags=2)
            self.canvas_pre.configure(image=self.cv2_to_ctk(img_m, 700, 300), text="")
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            res = cv2.warpPerspective(img_r, H, (img_l.shape[1] + img_r.shape[1], img_l.shape[0]))
            res[0:img_l.shape[0], 0:img_l.shape[1]] = img_l
            self.canvas_pos.configure(image=self.cv2_to_ctk(res, 700, 300), text="")
        else:
            messagebox.showerror("Erro", "Matches insuficientes.")

if __name__ == "__main__":
    app = VisionApp()
    app.mainloop()