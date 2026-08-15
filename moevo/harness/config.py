"""Configuration constants for the custom harness agent.

All tuneable parameters live here so the meta-improver can adjust them
without touching tool or loop code.
"""

# ── Model defaults ──────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 16384

# ── Loop limits ─────────────────────────────────────────────────────────────

MAX_ITERATIONS = 30  # max tool-call rounds per task
DOOM_LOOP_WINDOW = 6  # track last N call signatures
DOOM_LOOP_NUDGES = 3  # nudges before hard-terminating

# ── Tool limits ─────────────────────────────────────────────────────────────

BASH_TIMEOUT = 120  # seconds per shell command
MAX_BASH_OUTPUT = 15000  # chars of stdout/stderr to keep
MAX_FILE_READ = 60000  # chars before truncating file reads
MAX_GREP_RESULTS = 50  # max matches returned by grep
MAX_GLOB_RESULTS = 100  # max files returned by glob

# ── Retry ───────────────────────────────────────────────────────────────────

API_MAX_RETRIES = 3
API_BACKOFF_BASE = 2  # seconds; doubles each retry
