from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

FEEDBACK_PATH = Path("data/feedback/feedback_log.csv")


def _append_feedback(mode: str, predicted_label: str, correct_label: str, is_correct: str, confidence: float) -> str:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp,mode,predicted_label,correct_label,is_correct,confidence\n"
    if not FEEDBACK_PATH.exists():
        FEEDBACK_PATH.write_text(header, encoding="utf-8")

    row = f"{datetime.now(timezone.utc).isoformat()},{mode},{predicted_label},{correct_label},{is_correct},{confidence:.4f}\n"
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(row)
    return "Feedback saved. Thank you!"


def _predict_placeholder(video_path: str | None, mode: str):
    if not video_path:
        return "No video uploaded", 0.0
    # TODO: integrate real model inference for both dynamic and static modes.
    return f"TODO_PREDICTION_{mode.replace(' ', '_').upper()}", 0.0


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AnsabI mo? - FSLR") as demo:
        gr.Markdown("# AnsabI mo?\nMilestone scaffold for Filipino Sign Language Recognition (FSLR).")
        mode = gr.Radio(["Dynamic Words/Phrases", "Static Alphabet"], value="Dynamic Words/Phrases", label="Mode")
        video = gr.Video(label="Upload short signing video")
        pred_btn = gr.Button("Predict")

        predicted_label = gr.Textbox(label="Prediction result")
        confidence = gr.Number(label="Confidence", precision=4)

        pred_btn.click(fn=_predict_placeholder, inputs=[video, mode], outputs=[predicted_label, confidence])

        gr.Markdown("## Feedback")
        correct_label = gr.Textbox(label="Correct label (if prediction is wrong)")
        is_correct = gr.Radio(["Correct", "Wrong"], value="Correct", label="Was prediction correct?")
        submit_feedback = gr.Button("Submit Feedback")
        feedback_status = gr.Textbox(label="Feedback status")

        submit_feedback.click(
            fn=_append_feedback,
            inputs=[mode, predicted_label, correct_label, is_correct, confidence],
            outputs=[feedback_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
