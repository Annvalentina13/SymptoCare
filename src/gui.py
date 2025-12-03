import os
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd

from PIL import Image, ImageTk

from src.history import append_prediction, HISTORY_PATH
import csv
from datetime import datetime

def append_prediction(age, gender, symptoms, symptom_count, classes, proba, top_idx):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    predicted = classes[top_idx[0]]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "symptom_count": symptom_count,
        "predicted_disease": predicted,
        "top1_prob": float(proba[top_idx[0]]),
        "top2_disease": classes[top_idx[1]] if len(top_idx) > 1 else "",
        "top2_prob": float(proba[top_idx[1]]) if len(top_idx) > 1 else 0.0,
        "top3_disease": classes[top_idx[2]] if len(top_idx) > 2 else "",
        "top3_prob": float(proba[top_idx[2]]) if len(top_idx) > 2 else 0.0,
    }

    file_exists = os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.joblib")
EDA_DIR = os.path.join(BASE_DIR, "reports", "eda_plots")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "prediction_history.csv")

print("Starting SymptoCare GUI...")
class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SymptoCare - Login")
        self.geometry("460x360")
        self.configure(bg="#f4f7fb")
        self.resizable(False, False)

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        # Centered card, enough height to show button
        card = tk.Frame(self, bg="white", bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor="center", width=380, height=230)

        title = tk.Label(
            card,
            text="SymptoCare Login",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            fg="#1f4f7b",
        )
        title.pack(pady=(12, 2))

        subtitle = tk.Label(
            card,
            text="Sign in to access the dashboard",
            font=("Segoe UI", 9),
            bg="white",
            fg="#7b7b7b",
        )
        subtitle.pack(pady=(0, 8))

        form = tk.Frame(card, bg="white")
        form.pack(pady=(0, 4), padx=24, fill="x")

        tk.Label(form, text="Username", font=("Segoe UI", 9), bg="white").pack(anchor="w")
        tk.Entry(
            form,
            textvariable=self.username_var,
            font=("Segoe UI", 10),
            bd=1,
            relief="solid",
        ).pack(fill="x", pady=(2, 6))

        tk.Label(form, text="Password", font=("Segoe UI", 9), bg="white").pack(anchor="w")
        tk.Entry(
            form,
            textvariable=self.password_var,
            font=("Segoe UI", 10),
            bd=1,
            relief="solid",
            show="*",
        ).pack(fill="x", pady=(2, 4))

        login_btn = tk.Button(
            card,
            text="Login",
            font=("Segoe UI", 10, "bold"),
            bg="#22b8a7",
            fg="white",
            bd=0,
            relief="flat",
            activebackground="#1fa495",
            activeforeground="white",
            command=self._on_login,
        )
        login_btn.pack(pady=(6, 8), ipadx=14, ipady=4)


    def _on_login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()

        if username == "admin" and password == "admin":
            self.destroy()
            app = MainApp(username=username)
            app.mainloop()
        else:
            messagebox.showerror("Login failed", "Invalid username or password.")


class MainApp(tk.Tk):
    def __init__(self, username: str):
        super().__init__()
        self.title("SymptoCare - Provider Dashboard")
        self.geometry("1280x720")
        self.minsize(1100, 650)
        self.configure(bg="#f4f7fb")

        self.username = username
        self.model = None

        self._load_model()
        self._build_layout()

    def _load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model error", f"Could not load model:\n{e}")

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = tk.Frame(self, bg="#101828")
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_rowconfigure(4, weight=1)

        logo = tk.Label(
            sidebar,
            text="SC",
            font=("Segoe UI", 20, "bold"),
            bg="#22b8a7",
            fg="white",
            width=3,
        )
        logo.grid(row=0, column=0, pady=(16, 24), padx=16)

        nav_items = [("Dashboard", 1), ("Predict", 2), ("Analytics", 3), ("History", 4)]
        self.nav_buttons = []
        for i, (text, tab_index) in enumerate(nav_items, start=1):
            btn = tk.Button(
                sidebar,
                text=text,
                anchor="w",
                font=("Segoe UI", 10),
                bg="#101828",
                fg="white",
                activebackground="#1d2939",
                activeforeground="white",
                bd=0,
                relief="flat",
                command=lambda idx=tab_index: self._select_tab(idx),
            )
            btn.grid(
                row=i,
                column=0,
                sticky="ew",
                padx=12,
                pady=2,
                ipadx=12,
                ipady=4,
            )
            self.nav_buttons.append(btn)

        user_label = tk.Label(
            sidebar,
            text=f"Logged in as\n{self.username}",
            font=("Segoe UI", 9),
            bg="#101828",
            fg="#9ca3af",
            justify="left",
        )
        user_label.grid(row=5, column=0, sticky="sw", padx=16, pady=16)

        # Main area
        main_frame = tk.Frame(self, bg="#f4f7fb")
        main_frame.grid(row=0, column=1, sticky="nsew")

        # Top bar
        topbar = tk.Frame(main_frame, bg="#ffffff")
        topbar.pack(fill="x")

        title_label = tk.Label(
            topbar,
            text="Provider Dashboard",
            font=("Segoe UI", 16, "bold"),
            bg="#ffffff",
            fg="#1f4f7b",
        )
        title_label.pack(side="left", padx=24, pady=12)

        # Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#f4f7fb", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 10),
            padding=(16, 6),
            background="#e5edf6",
        )
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])

        self.dashboard_tab = tk.Frame(self.notebook, bg="#f4f7fb")
        self.predict_tab = tk.Frame(self.notebook, bg="#f4f7fb")
        self.analytics_tab = tk.Frame(self.notebook, bg="#f4f7fb")
        self.history_tab = tk.Frame(self.notebook, bg="#f4f7fb")

        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.predict_tab, text="Predict")
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.notebook.add(self.history_tab, text="History")

        self._build_dashboard_tab()
        self._build_predict_tab()
        self._build_analytics_tab()
        self._build_history_tab()
        self._refresh_analytics()

    def _select_tab(self, index: int):
        self.notebook.select(index - 1)

    # Dashboard tab
    def _build_dashboard_tab(self):
        cards_frame = tk.Frame(self.dashboard_tab, bg="#f4f7fb")
        cards_frame.pack(fill="x", padx=8, pady=(12, 4))

        def make_card(parent, title, value, color):
            frame = tk.Frame(parent, bg="white", bd=0, highlightthickness=0)
            frame.pack(side="left", padx=8, pady=4, fill="x", expand=True)
            tk.Label(
                frame,
                text=title,
                font=("Segoe UI", 9),
                bg="white",
                fg="#6b7280",
            ).pack(anchor="w", padx=12, pady=(10, 2))
            tk.Label(
                frame,
                text=value,
                font=("Segoe UI", 18, "bold"),
                bg="white",
                fg=color,
            ).pack(anchor="w", padx=12, pady=(0, 10))

        make_card(cards_frame, "Total Patients", "5000", "#22b8a7")
        make_card(cards_frame, "Diseases", "30", "#0ea5e9")
        make_card(cards_frame, "Model Accuracy", "3.2%", "#f97316")

        body = tk.Frame(self.dashboard_tab, bg="#f4f7fb")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # show disease_counts.png in the center
        img_path = os.path.join(EDA_DIR, "disease_counts.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((800, 350), Image.LANCZOS)
            self.dashboard_img = ImageTk.PhotoImage(img)
            lbl = tk.Label(body, image=self.dashboard_img, bg="#f4f7fb")
            lbl.pack(pady=10)
        else:
            placeholder = tk.Label(
                body,
                text="Run: python -m src.eda to generate dashboard charts.",
                font=("Segoe UI", 11),
                bg="#f4f7fb",
                fg="#6b7280",
            )
            placeholder.pack(pady=40)


    # Predict tab
        # Predict tab
    def _build_predict_tab(self):
        container = tk.Frame(self.predict_tab, bg="#f4f7fb")
        container.pack(fill="both", expand=True, padx=12, pady=12)

        form_frame = tk.Frame(container, bg="white")
        form_frame.pack(side="left", fill="y", padx=(0, 8), pady=0, ipadx=10, ipady=10)

        result_frame = tk.Frame(container, bg="white")
        result_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(8, 0),
            pady=0,
            ipadx=10,
            ipady=10,
        )

        tk.Label(
            form_frame,
            text="Patient Details",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#1f4f7b",
        ).pack(anchor="w", pady=(4, 8))

        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar(value="Male")
        self.symptom_text = tk.Text(form_frame, height=5, width=40)

        tk.Label(form_frame, text="Age", font=("Segoe UI", 9), bg="white").pack(
            anchor="w", pady=(4, 0)
        )
        tk.Entry(
            form_frame,
            textvariable=self.age_var,
            font=("Segoe UI", 10),
            bd=1,
            relief="solid",
        ).pack(fill="x", pady=(2, 6))

        tk.Label(form_frame, text="Gender", font=("Segoe UI", 9), bg="white").pack(
            anchor="w", pady=(4, 0)
        )
        gender_combo = ttk.Combobox(
            form_frame,
            textvariable=self.gender_var,
            values=["Male", "Female", "Other"],
            state="readonly",
        )
        gender_combo.pack(fill="x", pady=(2, 6))

        tk.Label(
            form_frame,
            text="Symptoms (comma-separated)",
            font=("Segoe UI", 9),
            bg="white",
        ).pack(anchor="w", pady=(4, 0))
        self.symptom_text.pack(fill="x", pady=(2, 6))

        predict_btn = tk.Button(
            form_frame,
            text="Predict Disease",
            font=("Segoe UI", 10, "bold"),
            bg="#22b8a7",
            fg="white",
            bd=0,
            relief="flat",
            activebackground="#1fa495",
            command=self._on_predict,
        )
        predict_btn.pack(pady=(8, 4), fill="x")

        tk.Label(
            result_frame,
            text="Prediction Result",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#1f4f7b",
        ).pack(anchor="w", pady=(4, 4))

        self.pred_label = tk.Label(
            result_frame,
            text="No prediction yet.",
            font=("Segoe UI", 11),
            bg="white",
            fg="#374151",
            wraplength=500,
            justify="left",
        )
        self.pred_label.pack(anchor="w", pady=(4, 4))

    def _on_predict(self):
        if self.model is None:
            messagebox.showerror("Model error", "Model not loaded.")
            return

        try:
            age = float(self.age_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter a valid age.")
            return

        gender = self.gender_var.get()
        symptoms_text = self.symptom_text.get("1.0", "end").strip()
        if not symptoms_text:
            messagebox.showerror("Input error", "Please enter at least one symptom.")
            return

        symptoms_list = [s.strip() for s in symptoms_text.split(",") if s.strip()]
        symptom_count = len(symptoms_list)

        df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Symptoms": ", ".join(symptoms_list),
                    "Symptom_Count": symptom_count,
                }
            ]
        )

        try:
            proba = self.model.predict_proba(df)[0]
            classes = self.model.classes_
            top_idx = proba.argsort()[::-1][:3]

            # log to history
            append_prediction(
                age=age,
                gender=gender,
                symptoms=", ".join(symptoms_list),
                symptom_count=symptom_count,
                classes=classes,
                proba=proba,
                top_idx=top_idx,
            )

            lines = []
            lines.append(
                f"Top prediction: {classes[top_idx[0]]} "
                f"({proba[top_idx[0]]*100:.1f}% probability)"
            )

            lines.append("")
            lines.append("Top 3 diseases:")
            for i in top_idx:
                lines.append(f"• {classes[i]} - {proba[i]*100:.1f}%")

            lines.append(
                "\nNote: This is an educational tool, "
                "not a medical diagnosis. Always consult a doctor."
            )

            self.pred_label.config(text="\n".join(lines))
        except Exception as e:
            messagebox.showerror("Prediction error", f"Could not predict:\n{e}")

    # Analytics tab (placeholder for now)
    def _build_analytics_tab(self):
        frame = tk.Frame(self.analytics_tab, bg="#f4f7fb")
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        tk.Label(
            frame,
            text="Prediction History Analytics",
            font=("Segoe UI", 12, "bold"),
            bg="#f4f7fb",
            fg="#1f4f7b",
        ).pack(anchor="w", pady=(4, 8))

        self.analytics_canvas = tk.Label(
            frame,
            text="No prediction history yet. Make some predictions first.",
            font=("Segoe UI", 11),
            bg="#f4f7fb",
            fg="#6b7280",
            wraplength=800,
            justify="left",
        )
        self.analytics_canvas.pack(pady=20)


    def _refresh_analytics(self):
        # Show chart of predictions per disease from history
        if not os.path.exists(HISTORY_PATH):
            self.analytics_canvas.config(
                text="No prediction history yet. Make some predictions first.",
                image="",
            )
            return

        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        df = pd.read_csv(HISTORY_PATH)
        if df.empty:
            self.analytics_canvas.config(
                text="No prediction history yet. Make some predictions first.",
                image="",
            )
            return

        counts = df["predicted_disease"].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="Blues_r")
        ax.set_xlabel("Number of predictions")
        ax.set_ylabel("Disease")
        ax.set_title("Top predicted diseases (history)")
        fig.tight_layout()

        hist_plot_path = os.path.join(EDA_DIR, "history_predictions.png")
        os.makedirs(EDA_DIR, exist_ok=True)
        fig.savefig(hist_plot_path, dpi=120)
        plt.close(fig)

        img = Image.open(hist_plot_path)
        img = img.resize((800, 350), Image.LANCZOS)
        self.analytics_img = ImageTk.PhotoImage(img)
        self.analytics_canvas.config(image=self.analytics_img, text="")


    # History tab (placeholder)
    def _build_history_tab(self):
        frame = tk.Frame(self.history_tab, bg="#f4f7fb")
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        label = tk.Label(
            frame,
            text="Prediction history table will be added here.",
            font=("Segoe UI", 11),
            bg="#f4f7fb",
            fg="#6b7280",
            wraplength=800,
            justify="left",
        )
        label.pack(pady=40)


if __name__ == "__main__":
    login = LoginWindow()
    login.mainloop()

