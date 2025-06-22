# MODA_CK: Crisis Companion Q&A Bot

## Quickstart (under 30 min)

1. `python -m venv .venv && source .venv/bin/activate` (or `Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your Twilio/Together keys
4. Run `python app.py`
5. In a new terminal: `ngrok http 5050` (copy the https URL)
6. Paste ngrok URL as your webhook in Twilio Console (Messaging > Sandbox)
7. Drop PDFs into `pdf_knowledge_base/` (auto-ingested within 1 min)
8. Send a WhatsApp/SMS to your Twilio number, ask a question
9. Check logs for errors and LLM latency
10. (Optional) Test with `bash test.sh`

## Notes
- Only free-tier/open-source services used
- PDFs are hot-reloaded (watchdog)
- LLM: Together-AI (if available) or Hugging Face public API