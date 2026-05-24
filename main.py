import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Загружаем переменные из .env (локально). На Railway переменные берутся из Variables.
load_dotenv()

# Langflow config
LANGFLOW_URL = (os.getenv("LANGFLOW_URL") or "http://127.0.0.1:7860").rstrip("/")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID") or ""
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY") or ""

LANGFLOW_INPUT_TYPE = os.getenv("LANGFLOW_INPUT_TYPE") or "chat"
LANGFLOW_OUTPUT_TYPE = os.getenv("LANGFLOW_OUTPUT_TYPE") or "chat"

# CORS config (для Lovable/браузера)
LOVEABLE_ORIGIN = os.getenv("LOVEABLE_ORIGIN") or ""  # например: https://happy-multiply-box.lovable.app
CORS_ALLOW_ALL = (os.getenv("CORS_ALLOW_ALL") or "").lower() in ("1", "true", "yes")

missing: List[str] = []
for k, v in [
    ("LANGFLOW_URL", LANGFLOW_URL),
    ("LANGFLOW_FLOW_ID", LANGFLOW_FLOW_ID),
    ("LANGFLOW_API_KEY", LANGFLOW_API_KEY),
]:
    if not v:
        missing.append(k)

if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


# --- HTTP client ---
# trust_env=False — игнорируем системные прокси (часто ломают локальные/railway вызовы)
timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)
client = httpx.AsyncClient(timeout=timeout, trust_env=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="Langflow FastAPI Proxy", lifespan=lifespan)

# --- CORS ---
# В проде лучше строго разрешать только домен Lovable.
# Для быстрой отладки можно включить CORS_ALLOW_ALL=true
if CORS_ALLOW_ALL:
    allow_origins = ["*"]
else:
    allow_origins = [LOVEABLE_ORIGIN] if LOVEABLE_ORIGIN else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class MultiplyRequest(BaseModel):
    numbers: List[float] = Field(min_length=2)
    session_id: Optional[str] = None


def _extract_text_from_langflow(resp_json: Dict[str, Any]) -> Optional[str]:
    try:
        return resp_json["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except (KeyError, IndexError, TypeError):
        return None


def _make_auth_headers() -> List[Dict[str, str]]:
    # Langflow иногда ожидает один из двух вариантов авторизации
    return [
        {"Authorization": f"Bearer {LANGFLOW_API_KEY}"},
        {"x-api-key": LANGFLOW_API_KEY},
    ]


async def _run_langflow(input_value: str, session_id: str) -> Tuple[Dict[str, Any], str]:
    url = f"{LANGFLOW_URL}/api/v1/run/{LANGFLOW_FLOW_ID}"

    payload: Dict[str, Any] = {
        "input_value": input_value,
        "input_type": LANGFLOW_INPUT_TYPE,
        "output_type": LANGFLOW_OUTPUT_TYPE,
        "session_id": session_id,
        "tweaks": None,
    }

    last_status: Optional[int] = None
    last_text: Optional[str] = None

    for auth in _make_auth_headers():
        headers = {"Content-Type": "application/json", **auth}

        try:
            r = await client.post(url, json=payload, headers=headers)
            last_status = r.status_code
            last_text = r.text

            # если не подошёл способ авторизации — пробуем следующий
            if r.status_code in (401, 403):
                continue

            r.raise_for_status()
            return r.json(), ("bearer" if "Authorization" in auth else "x-api-key")

        except httpx.HTTPStatusError as e:
            # Langflow вернул 4xx/5xx — пробрасываем как есть
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text,
            ) from e

        except httpx.ReadTimeout as e:
            raise HTTPException(
                status_code=502,
                detail=f"Langflow request error: ReadTimeout {repr(e)}",
            ) from e

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Langflow request error: {type(e).__name__} {repr(e)}",
            ) from e

    # если оба способа auth дали 401/403
    raise HTTPException(
        status_code=502,
        detail=(
            "Langflow auth failed (tried Bearer and x-api-key). "
            f"Last status={last_status}. Last response={last_text}"
        ),
    )


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "up"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/multiply")
async def multiply(req: MultiplyRequest) -> Dict[str, Any]:
    input_value = " * ".join(str(x) for x in req.numbers)

    resp_json, auth_used = await _run_langflow(
        input_value=input_value,
        session_id=req.session_id or "multiply-session",
    )

    return {
        "input": input_value,
        "auth_used": auth_used,
        "result_text": _extract_text_from_langflow(resp_json),
        "raw": resp_json,
    }
