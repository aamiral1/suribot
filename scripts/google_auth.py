"""
One-time script to authorise Google Calendar access for suribot.

Prerequisites:
  1. Create a Google Cloud project at console.cloud.google.com
  2. Enable the Google Calendar API
  3. Create an OAuth 2.0 Client ID (type: Desktop app)
  4. Download the JSON and save as scripts/client_secret.json
  5. Run: python scripts/google_auth.py

The script opens a browser window for OAuth2 consent, then saves
google_token.json to the project root. The main app will refresh
this token automatically — you only need to run this once.
"""

import os
import sys
import argparse

# Allow running from project root: python scripts/google_auth.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def main():
    parser = argparse.ArgumentParser(
        description="Authorise Google Calendar access for suribot"
    )
    parser.add_argument(
        "--secrets",
        default=os.getenv("GOOGLE_CLIENT_SECRETS_PATH", "scripts/client_secret.json"),
        help="Path to client_secret.json (downloaded from Google Cloud Console)",
    )
    parser.add_argument(
        "--token-path",
        default=os.getenv("GOOGLE_TOKEN_PATH", "google_token.json"),
        help="Where to write the token (default: google_token.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.secrets):
        print(f"\nError: client secrets file not found at '{args.secrets}'")
        print("\nTo fix this:")
        print("  1. Go to console.cloud.google.com")
        print("  2. APIs & Services → Credentials → OAuth 2.0 Client IDs")
        print("  3. Download JSON → save as scripts/client_secret.json")
        sys.exit(1)

    print("Opening browser for Google OAuth2 consent...")
    print("(If a 'This app isn't verified' warning appears, click Advanced → Go to app)\n")

    flow = InstalledAppFlow.from_client_secrets_file(args.secrets, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(args.token_path, "w") as f:
        f.write(creds.to_json())

    print(f"\nToken saved to: {args.token_path}")
    print(f"Token expires: {creds.expiry} (auto-refreshes using stored refresh_token)")

    # Smoke test: verify the token works by fetching the calendar name
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    service = build("calendar", "v3", credentials=creds)
    try:
        result = service.calendarList().get(calendarId=calendar_id).execute()
        print(f"Connected to calendar: '{result.get('summary', calendar_id)}'")
        print("\nSetup complete. The app is now authorised to read/write this calendar.")
    except Exception as e:
        print(f"\nWarning: token saved but smoke test failed: {e}")
        print("The calendar ID may be wrong. Check GOOGLE_CALENDAR_ID in .env")


if __name__ == "__main__":
    main()
