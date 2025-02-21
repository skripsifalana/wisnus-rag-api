# app/config/rotating_llm_wrapper.py

from ragas.llms import LangchainLLMWrapper
from app.config.llm2 import get_next_llm, gemini_keys

class RotatingLangchainLLMWrapper(LangchainLLMWrapper):
    def __init__(self, llm_holder, run_config=None):
        # Inisialisasi dengan llm yang ada di llm_holder
        super().__init__(llm_holder.llm, run_config=run_config)
        self.llm_holder = llm_holder

    async def _acall(self, messages):
        """Override fungsi _acall untuk menangani error 429 dengan rotasi API key."""
        attempts = 0
        max_attempts = len(gemini_keys)
        while attempts < max_attempts:
            try:
                # Panggil LLM menggunakan instance yang ada di llm_holder
                return await self.llm_holder.llm.ainvoke(messages)
            except Exception as e:
                if "429" in str(e):
                    attempts += 1
                    # Jika error 429, rotasi API key dengan mendapatkan instance LLM baru
                    self.llm_holder.llm = await get_next_llm()
                    print(f"Rotasi API key: percobaan {attempts}/{max_attempts}")
                else:
                    raise e
        raise Exception("Semua API key telah digunakan karena batas rate limit.")
