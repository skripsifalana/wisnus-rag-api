import os
from typing import Tuple, Any
from app.types.main_types import KeyState, ServiceResponse

async def initialize_key_state() -> KeyState:
    # Ambil API key dari environment (GEMINI_API_KEY_1 hingga GEMINI_API_KEY_10)
    keys = []
    for i in range(1, 11):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    return KeyState(current_index=0, keys=keys)

async def get_current_key(key_state: KeyState) -> ServiceResponse[Tuple[str, KeyState]]:
    try:
        if not key_state.keys:
            raise Exception("No API keys available")
        api_key = key_state.keys[key_state.current_index]
        # Kembalikan API key saat ini dan state (tanpa perubahan)
        return ServiceResponse(success=True, data=(api_key, key_state), metadata={})
    except Exception as e:
        return ServiceResponse(success=False, error=str(e), metadata={})

async def handle_error(key_state: KeyState, error: Any) -> ServiceResponse[Tuple[None, KeyState]]:
    try:
        # Pada error, rotasi ke API key berikutnya (dengan wrap-around)
        if key_state.keys:
            key_state.current_index = (key_state.current_index + 1) % len(key_state.keys)
        else:
            key_state.current_index = 0
        return ServiceResponse(success=True, data=(None, key_state), metadata={})
    except Exception as e:
        return ServiceResponse(success=False, error=str(e), metadata={})

async def reset_counters(key_state: KeyState) -> ServiceResponse[KeyState]:
    try:
        key_state.current_index = 0
        return ServiceResponse(success=True, data=key_state, metadata={})
    except Exception as e:
        return ServiceResponse(success=False, error=str(e), metadata={})
