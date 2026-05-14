"""Flask REST API for audio, video, and final multimodal predictions."""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.audio.predict_audio import predict_audio
from backend.fusion.predict_fusion import predict_fusion
from backend.utils.io import UPLOADS_DIR, ensure_dirs
from backend.video.predict_video import predict_video

ALLOWED_AUDIO = {"wav", "mp3"}
ALLOWED_VIDEO = {"mp4"}
SESSION_RESULTS: dict[str, dict] = {}


def create_app() -> Flask:
    """Create Flask app for frontend integration."""
    ensure_dirs()
    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    def save_upload(field: str, allowed: set[str], subdir: str) -> tuple[str | None, Path | None, tuple | None]:
        if field not in request.files:
            return None, None, (jsonify({"error": f"Missing file field '{field}'"}), 400)
        file = request.files[field]
        if not file.filename:
            return None, None, (jsonify({"error": "Empty filename"}), 400)
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in allowed:
            return None, None, (jsonify({"error": f"Unsupported file type '.{ext}'. Allowed: {sorted(allowed)}"}), 400)
        session_id = request.form.get("session_id") or str(uuid.uuid4())
        upload_dir = UPLOADS_DIR / subdir
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{session_id}_{secure_filename(file.filename)}"
        path = upload_dir / filename
        file.save(path)
        return session_id, path, None

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict-audio")
    def predict_audio_endpoint():
        session_id, path, error = save_upload("file", ALLOWED_AUDIO, "audio")
        if error:
            return error
        try:
            result = predict_audio(path)
            SESSION_RESULTS.setdefault(session_id, {})["audio"] = result
            return jsonify({"session_id": session_id, "audio_prediction": result})
        except Exception as exc:  # Flask endpoint should return JSON errors.
            return jsonify({"error": str(exc)}), 500

    @app.post("/predict-video")
    def predict_video_endpoint():
        session_id, path, error = save_upload("file", ALLOWED_VIDEO, "video")
        if error:
            return error
        try:
            result = predict_video(path)
            SESSION_RESULTS.setdefault(session_id, {})["video"] = result
            response = {"session_id": session_id, "video_prediction": result}
            if "audio" in SESSION_RESULTS[session_id]:
                response["final_prediction"] = predict_fusion(SESSION_RESULTS[session_id]["audio"], result)
            return jsonify(response)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/predict-final")
    def predict_final_endpoint():
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id") or request.form.get("session_id")
        if not session_id or session_id not in SESSION_RESULTS:
            return jsonify({"error": "Unknown or missing session_id"}), 400
        session = SESSION_RESULTS[session_id]
        if "audio" not in session or "video" not in session:
            return jsonify({"error": "Both audio and video predictions are required before fusion"}), 400
        try:
            result = predict_fusion(session["audio"], session["video"])
            session["final"] = result
            return jsonify({"session_id": session_id, "final_prediction": result})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
