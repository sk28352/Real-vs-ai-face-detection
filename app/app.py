import os
import random
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# configurations
N_ROUNDS = 10
IMG_DIR = "test_images"
MODEL_PATH = "best_model.keras"

# Load model + images once at startup
model = tf.keras.models.load_model(MODEL_PATH)

files  = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))
images = [np.array(Image.open(os.path.join(IMG_DIR, f)).convert("RGB")) for f in files]
labels = [1 if f.startswith("real") else 0 for f in files]          # 1 = Real, 0 = AI

# MobileNetV3 expects raw [0,255] pixels; sigmoid output = P(Real)
scores = model.predict(np.stack(images).astype("float32"), verbose=0).ravel()

def label_name(y):
    return "Real 👤" if y == 1 else "AI-generated 🤖"

# ---------------- UI helpers ----------------
def banner(text, color):
    return (f"<div style='padding:8px 10px;border-radius:10px;background:{color};"
            f"color:#111 !important;font-size:14px;text-align:center;margin-bottom:6px'>"
            f"<span style='color:#111 !important'>{text}</span></div>")

def round_text(s):
    current = s["round"] if s["answered"] else s["round"] + 1
    return f"### Round {current} / {N_ROUNDS}"


def score_html(s):
    done = s["round"]
    def box(title, pts, color):
        acc = f"{100 * pts / done:.0f}%" if done else "–"
        dark = "color:#111 !important"
        return (f"<div style='flex:1;padding:8px;border-radius:10px;background:{color};text-align:center;{dark}'>"
                f"<div style='font-size:13px;{dark}'>{title}</div>"
                f"<div style='font-size:22px;font-weight:700;{dark}'>{pts} / {done}</div>"
                f"<div style='font-size:12px;{dark}'>accuracy {acc}</div></div>")
    return ("<div style='display:flex;gap:8px;margin-bottom:6px'>"
            + box("🧑 You", s["human"], "#dbeafe") + box("🤖 Model", s["model"], "#fde68a") + "</div>")

# ---------------- Game logic ----------------
# Every handler returns: state, image, round, score, feedback, btn_real, btn_ai, btn_next
def show_round(s):
    s["answered"] = False
    idx = s["order"][s["round"]]
    return (s, images[idx], round_text(s), score_html(s),
            banner("Is this face <b>Real</b> or <b>AI-generated</b>?", "#f3f4f6"),
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False))

def new_game():
    s = {"order": random.sample(range(len(files)), N_ROUNDS),
         "round": 0, "human": 0, "model": 0, "answered": False}
    return show_round(s)

def guess(choice, s):
    if s is None or s["answered"]:                       # ignore double clicks
        return (s,) + tuple(gr.update() for _ in range(7))

    idx = s["order"][s["round"]]
    truth, p = labels[idx], float(scores[idx])
    model_guess = int(p > 0.5)
    conf = p if model_guess == 1 else 1 - p
    human_ok, model_ok = (choice == truth), (model_guess == truth)

    s["human"] += int(human_ok)
    s["model"] += int(model_ok)
    s["round"] += 1
    s["answered"] = True

    fb = banner(f"Answer: <b>{label_name(truth)}</b><br>"
                f"You said {label_name(choice)} {'✅' if human_ok else '❌'} &nbsp;|&nbsp; "
                f"Model said {label_name(model_guess)} ({conf:.0%} sure) {'✅' if model_ok else '❌'}",
                "#bbf7d0" if human_ok else "#fecaca")

    finished = s["round"] >= N_ROUNDS
    if finished:
        if s["human"] > s["model"]:
            result = "🎉🎉 You beat the AI detector! 🎉🎉"
        elif s["human"] < s["model"]:
            result = "🤖 The model wins this time. Rematch?"
        else:
            result = "🤝 It's a tie!"
        fb += banner(f"<b>Game over!</b> You {s['human']}/{N_ROUNDS} vs Model {s['model']}/{N_ROUNDS}"
                     f"<br><span style='font-size:22px'>{result}</span><br>"
                     f"Press <b>New Game</b> to play again.", "#e9d5ff")

    return (s, gr.update(), round_text(s), score_html(s), fb,
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=not finished))

def next_round(s):
    if s is None or not s["answered"] or s["round"] >= N_ROUNDS:
        return (s,) + tuple(gr.update() for _ in range(7))
    return show_round(s)

# ---------------- Layout ----------------
CSS = """
.gradio-container {max-width: 820px !important; margin: 0 auto !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Real or AI? Human vs. Model") as demo:
    gr.Markdown("## 🤖 Real or AI? Human vs. Model\n"
                f"Is the face **real** or **AI-generated**? A fine-tuned MobileNetV3 answers the "
                f"same image. Best of {N_ROUNDS} rounds wins!")
    state = gr.State()
    with gr.Row(equal_height=False):
        # Left: image + guess buttons
        with gr.Column(scale=1, min_width=260):
            img = gr.Image(show_label=False, height=260, interactive=False)
            with gr.Row():
                btn_real = gr.Button("👤 Real", variant="primary", interactive=False)
                btn_ai   = gr.Button("🤖 AI-Generated", variant="primary", interactive=False)
        # Right: round, scores, feedback, navigation
        with gr.Column(scale=1, min_width=260):
            round_md = gr.Markdown()
            score    = gr.HTML()
            feedback = gr.HTML()
            with gr.Row():
                btn_next = gr.Button("➡️ Next Image", size="sm", interactive=False)
                btn_new  = gr.Button("🔄 New Game", size="sm", variant="secondary")

    outputs = [state, img, round_md, score, feedback, btn_real, btn_ai, btn_next]
    btn_new.click(new_game, None, outputs)
    btn_real.click(lambda s: guess(1, s), state, outputs)
    btn_ai.click(lambda s: guess(0, s), state, outputs)
    btn_next.click(next_round, state, outputs)
    demo.load(new_game, None, outputs)

if __name__ == "__main__":
    demo.launch()