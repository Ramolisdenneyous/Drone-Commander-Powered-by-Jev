from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and not url.startswith("postgresql+psycopg://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://drone:drone@localhost:5436/drone_commander"
    content_root: str = "content"
    content_version: str = "vs-0.1.0"
    llm_provider: str = "openai"
    llm_external_enabled: bool = True
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model_tactical: str = "gpt-5.6-luna"
    llm_model_radio: str = "gpt-5.6-luna"
    llm_decision_timeout_sec: float = 20.0
    llm_radio_timeout_sec: float = 8.0
    # Tactical decisions may use OpenAI tool calls or TypeSafe Jev's typed choice API.
    # Radio remains a separate generative concern.
    tactical_provider: str = "openai"
    typesafe_api_key: str = ""
    typesafe_base_url: str = "https://api.typesafe.ai"
    typesafe_model: str = "jev-latest"
    jev_decision_timeout_sec: float = 3.0
    jev_min_confidence: float = 0.60
    jev_min_margin: float = 0.10
    # auto: template while Jev owns tactics, OpenAI while Luna owns tactics.
    # template and openai are explicit overrides.
    radio_provider: str = "auto"
    # Shadow asks Jev for a choice and logs it, then commits the deterministic fallback.
    tactical_shadow: bool = False
    log_level: str = "INFO"
    # metadata = hashes/summaries only; full_diagnostic = include truncated prompt/response text
    artifact_retention_mode: str = "metadata"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def model_post_init(self, __context) -> None:
        self.database_url = _normalize_database_url(self.database_url)


settings = Settings()
