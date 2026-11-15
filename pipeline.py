import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pymupdf
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# =====================================================================
# 1. CONFIG LOADER
# =====================================================================

def _load_env_file(env_path: str = ".env") -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def get_or_prompt_api_key(key_name: str, display_name: str) -> str:
    api_key = os.environ.get(key_name)
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        return api_key.strip()
    
    if not sys.stdin.isatty():
        print(f"❌ Error: {display_name} not found in environment and running non-interactively.")
        sys.exit(1)
        
    print(f"⚠️  {display_name} not found in .env.")
    user_input = input(f"Please enter your {display_name}: ").strip()
    if not user_input:
        sys.exit(f"❌ Error: {display_name} cannot be empty.")
    return user_input

def load_all_api_keys():
    _load_env_file(".env")
    return {
        "GEMINI_API_KEY": get_or_prompt_api_key("GEMINI_API_KEY", "Gemini API Key"),
        "GROQ_API_KEY": get_or_prompt_api_key("GROQ_API_KEY", "Groq API Key"),
        "OPENROUTER_API_KEY": get_or_prompt_api_key("OPENROUTER_API_KEY", "OpenRouter API Key")
    }

# =====================================================================
# 2. FREE LLM ROUTER (WITH 404 STABILITY FILTERS & SPEED RANKING)
# =====================================================================

DEFAULT_CACHE_FILE = ".free_models_cache.json"

PROVIDER_RATE_INFO = {
    "ollama": ("Unlimited (Local)", "100% Free (Local)"),
    "gemini": ("~300-900 RPH (Flash)", "Free tier (Pro=0 quota)"),
    "groq": ("~1,800 RPH (30 RPM)", "Free tier (RPD capped)"),
    "openrouter": ("~1,200 RPH (50 RPD)", "100% Free (:free)"),
    "nvidia": ("~2,400 RPH (40 RPM)", "Free dev credits"),
    "huggingface": ("~100-300 RPH", "Free monthly quota"),
}

def _estimate_model_tps(provider: str, model_id: str) -> float:
    mid = model_id.lower()
    if provider == "groq":
        if any(k in mid for k in ("8b", "instant", "gemma", "small")): return 750.0
        if "70b" in mid or "versatile" in mid: return 320.0
        return 450.0
    if provider == "gemini":
        if "lite" in mid or "8b" in mid: return 240.0
        if "flash" in mid: return 175.0
        if "gemma" in mid: return 140.0
        return 85.0
    if provider == "nvidia":
        if any(k in mid for k in ("8b", "7b", "mini", "small", "nano")): return 145.0
        return 85.0
    if provider == "openrouter":
        if any(k in mid for k in ("flash", "8b", "7b", "mini", "small")): return 95.0
        return 55.0
    if provider == "huggingface":
        if any(k in mid for k in ("8b", "7b", "mini", "small")): return 75.0
        return 45.0
    if provider == "ollama":
        if any(k in mid for k in ("1b", "3b", "mini", "tiny")): return 65.0
        if any(k in mid for k in ("7b", "8b")): return 35.0
        return 18.0
    return 50.0

def _is_valid_key(val: Optional[str]) -> bool:
    return bool(val and val.strip() and not val.strip().startswith("your_"))

def _sort_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        models,
        key=lambda m: (
            m.get("input_limit", 0) + m.get("output_limit", 0),
            m.get("output_limit", 0),
            m.get("input_limit", 0),
        ),
        reverse=True,
    )

def _sort_models_by_speed(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        models,
        key=lambda m: (
            float(m.get("measured_tps") or m.get("est_tps") or _estimate_model_tps(m.get("provider", ""), m.get("id", ""))),
            m.get("output_limit", 0),
            m.get("input_limit", 0),
        ),
        reverse=True,
    )

def _encode_image(image_path: str) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type

class FreeLLMRouter:
    OPTIONS = {
        "1": "Local Ollama Models",
        "2": "Google Gemini (Free Tier)",
        "3": "Groq (Free Tier)",
        "4": "OpenRouter (Free Models)",
        "5": "Other Free APIs (NVIDIA NIM / Hugging Face)",
    }

    def __init__(self, force_refresh: bool = False, timeout: int = 90) -> None:
        keys = load_all_api_keys()
        for k, v in keys.items():
            os.environ[k] = v
        self.cache_path = Path(DEFAULT_CACHE_FILE)
        self.timeout = timeout
        self.selected_option: Optional[str] = None
        self.custom_model: Optional[Dict[str, str]] = None
        self.models_by_option: Dict[str, List[Dict[str, Any]]] = {}

        if not force_refresh and self.cache_path.exists():
            try:
                self.models_by_option = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.models_by_option = {}

        if not self.models_by_option or force_refresh:
            self.discover_and_cache_models()

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self.models_by_option, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[Router] Cache save warning: {exc}")

    def discover_and_cache_models(self) -> None:
        print("[Router] Discovering available free models (filtering out preview/404 endpoints)...")
        discovered = {
            "1": _sort_models(self._discover_ollama()),
            "2": _sort_models(self._discover_gemini()),
            "3": _sort_models(self._discover_groq()),
            "4": _sort_models(self._discover_openrouter()),
            "5": _sort_models(self._discover_other_free()),
        }
        self.models_by_option = discovered
        self._save_cache()
        print(f"[Router] Saved ranked stable model lists to {self.cache_path}")

    def _discover_ollama(self) -> List[Dict[str, Any]]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        models = []
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            if resp.ok:
                for item in resp.json().get("models", []):
                    name = item.get("name")
                    if name:
                        is_vision = any(k in name.lower() for k in ("llava", "vision", "vl", "minicpm", "gemma3", "qwen2.5vl"))
                        models.append({"id": name, "provider": "ollama", "input_limit": 8192, "output_limit": 4096, "vision": is_vision, "est_tps": _estimate_model_tps("ollama", name)})
        except Exception:
            pass
        return models

    def _discover_gemini(self) -> List[Dict[str, Any]]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not _is_valid_key(api_key): return []
        models = []
        try:
            resp = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("models", []):
                if "generateContent" not in item.get("supportedGenerationMethods", []): continue
                name = item.get("name", "").replace("models/", "")
                if any(bad in name.lower() for bad in ("embedding", "imagen", "veo", "tts", "aqa", "pro", "preview", "experimental")):
                    continue
                is_vision = "flash" in name.lower() or "gemma-3" in name.lower()
                models.append({
                    "id": name,
                    "provider": "gemini",
                    "input_limit": int(item.get("inputTokenLimit") or 0),
                    "output_limit": int(item.get("outputTokenLimit") or 0),
                    "vision": is_vision,
                    "est_tps": _estimate_model_tps("gemini", name)
                })
        except Exception as exc:
            print(f"[Router] Gemini discovery warning: {exc}")
        return models

    def _discover_groq(self) -> List[Dict[str, Any]]:
        api_key = os.getenv("GROQ_API_KEY")
        if not _is_valid_key(api_key): return []
        models = []
        try:
            resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                mid = item.get("id", "")
                if any(bad in mid.lower() for bad in ("whisper", "tts", "guard", "prompt-guard", "preview")): continue
                is_vision = any(k in mid.lower() for k in ("vision", "vl", "scout", "maverick", "qwen3"))
                models.append({
                    "id": mid,
                    "provider": "groq",
                    "input_limit": int(item.get("context_window") or 8192),
                    "output_limit": int(item.get("max_completion_tokens") or 8192),
                    "vision": is_vision,
                    "est_tps": _estimate_model_tps("groq", mid)
                })
        except Exception as exc:
            print(f"[Router] Groq discovery warning: {exc}")
        return models

    def _discover_openrouter(self) -> List[Dict[str, Any]]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not _is_valid_key(api_key): return []
        models = []
        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                mid = item.get("id", "")
                if any(bad in mid.lower() for bad in ("preview", "beta", "experimental")):
                    continue
                pricing = item.get("pricing", {})
                is_free = mid.endswith(":free") or mid == "openrouter/free" or (
                    str(pricing.get("prompt", "")) == "0" and str(pricing.get("completion", "")) == "0"
                )
                if not is_free: continue
                arch = item.get("architecture") or {}
                modalities = arch.get("input_modalities") or []
                is_vision = ("image" in modalities) or any(k in mid.lower() for k in ("vl", "vision", "flash", "gemini", "gemma-3", "pixtral", "molmo"))
                ctx = int(item.get("context_length") or 8192)
                out_lim = int((item.get("top_provider") or {}).get("max_completion_tokens") or min(ctx, 8192))
                models.append({
                    "id": mid,
                    "provider": "openrouter",
                    "input_limit": ctx,
                    "output_limit": out_lim,
                    "vision": is_vision,
                    "est_tps": _estimate_model_tps("openrouter", mid)
                })
        except Exception as exc:
            print(f"[Router] OpenRouter discovery warning: {exc}")
        return models

    def _discover_other_free(self) -> List[Dict[str, Any]]:
        models = []
        specs = [
            ("NVIDIA_API_KEY", "nvidia", "https://integrate.api.nvidia.com/v1/models", 32768, 4096),
            ("HF_API_KEY", "huggingface", "https://router.huggingface.co/v1/models", 16384, 4096),
        ]
        for env_var, prov, url, def_in, def_out in specs:
            key = os.getenv(env_var)
            if not _is_valid_key(key): continue
            try:
                resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=15)
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    mid = item.get("id", "")
                    if not mid or any(bad in mid.lower() for bad in ("embed", "rerank", "guard", "moderation", "parse", "preview")): continue
                    is_vision = any(k in mid.lower() for k in ("vl", "vision", "neva", "kosmos", "phi-3-vision", "phi-3.5-vision"))
                    models.append({
                        "id": mid,
                        "provider": prov,
                        "input_limit": def_in,
                        "output_limit": def_out,
                        "vision": is_vision,
                        "est_tps": _estimate_model_tps(prov, mid)
                    })
            except Exception:
                pass
        return models

    def _get_option_models(self, option: str, vision_only: bool = False) -> List[Dict[str, Any]]:
        pool = list(self.models_by_option.get(option, []))
        if vision_only:
            pool = [m for m in pool if m.get("vision", False)]
        return pool

    def _get_ranked_models(self, top_n: Optional[int] = 10, vision_only: bool = False) -> List[Dict[str, Any]]:
        all_models: List[Dict[str, Any]] = []
        for opt_key, items in self.models_by_option.items():
            for m in items:
                if vision_only and not m.get("vision"): continue
                entry = dict(m)
                entry["option"] = opt_key
                all_models.append(entry)
        ranked = _sort_models(all_models)
        return ranked[:top_n] if top_n else ranked

    def _get_speed_ranked_models(self, top_n: Optional[int] = 10, vision_only: bool = False) -> List[Dict[str, Any]]:
        all_models: List[Dict[str, Any]] = []
        for opt_key, items in self.models_by_option.items():
            for m in items:
                if vision_only and not m.get("vision"): continue
                entry = dict(m)
                entry["option"] = opt_key
                if "est_tps" not in entry:
                    entry["est_tps"] = _estimate_model_tps(entry.get("provider", ""), entry.get("id", ""))
                all_models.append(entry)
        ranked = _sort_models_by_speed(all_models)
        return ranked[:top_n] if top_n else ranked

    def compare_top_models(self, top_n: int = 10, vision_only: bool = False) -> None:
        ranked = self._get_ranked_models(top_n=top_n, vision_only=vision_only)
        if not ranked:
            print("[Router] No models cached yet. Add keys in .env and press 'r' to refresh.")
            return
        tag = "Vision " if vision_only else ""
        print(f"\n=== Top {len(ranked)} {tag}Models Side-by-Side (Sorted by Input + Output Token Limits) ===")
        header = f"{'#':<3} {'Opt':<4} {'Provider':<12} {'Model ID':<38} {'Vision':<7} {'In Limit':>10} {'Out Limit':>10} {'Est. Req/Hr':<21}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(ranked, 1):
            prov = m.get("provider", "?")
            rph, _ = PROVIDER_RATE_INFO.get(prov, ("Unknown", ""))
            mid = m.get("id", "")[:37]
            vis = "Yes" if m.get("vision") else "No"
            print(f"{i:<3} {m.get('option','?'):<4} {prov:<12} {mid:<38} {vis:<7} {m.get('input_limit',0):>10,} {m.get('output_limit',0):>10,} {rph:<21}")

    def benchmark_models_speed(self, top_n: int = 10, vision_only: bool = False) -> None:
        candidates = self._get_speed_ranked_models(top_n=top_n, vision_only=vision_only)
        if not candidates:
            print("[Router] No models available to benchmark.")
            return
        print(f"\n[Speed Benchmark] Running live output speed test across Top {len(candidates)} candidate models...")
        test_prompt = "Count from 1 to 40 separated by spaces and briefly state what an API router does in two sentences."
        for idx, m in enumerate(candidates, 1):
            prov, mid, opt_key = m["provider"], m["id"], m["option"]
            print(f"  -> ({idx}/{len(candidates)}) Testing {prov}:{mid} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                out = self._call_model(prov, mid, test_prompt, None, None, 0.1, 120)
                elapsed = max(time.perf_counter() - t0, 0.001)
                est_tokens = max(int(len(out.split()) * 1.35), 1)
                tps = round(est_tokens / elapsed, 1)
                latency_ms = int(elapsed * 1000)
                print(f"{tps} tok/s ({latency_ms} ms)")
                for item in self.models_by_option.get(opt_key, []):
                    if item.get("id") == mid and item.get("provider") == prov:
                        item["measured_tps"] = tps
                        item["latency_ms"] = latency_ms
            except Exception as exc:
                print(f"FAILED ({exc})")
        self._save_cache()

    def compare_models_by_speed(self, top_n: int = 10, vision_only: bool = False) -> None:
        ranked = self._get_speed_ranked_models(top_n=top_n, vision_only=vision_only)
        if not ranked:
            print("[Router] No models cached yet. Add keys in .env and press 'r' to refresh.")
            return
        tag = "Vision " if vision_only else ""
        print(f"\n=== Top {len(ranked)} Fastest {tag}Models Side-by-Side (Sorted by Output Speed) ===")
        header = f"{'#':<3} {'Opt':<4} {'Provider':<12} {'Model ID':<36} {'Speed (Tok/s)':>15} {'Source':<10} {'Out Limit':>10}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(ranked, 1):
            prov = m.get("provider", "?")
            mid = m.get("id", "")[:35]
            if m.get("measured_tps"):
                speed_val = float(m["measured_tps"])
                src = "Measured"
            else:
                speed_val = float(m.get("est_tps") or _estimate_model_tps(prov, m.get("id", "")))
                src = "Estimated"
            print(f"{i:<3} {m.get('option','?'):<4} {prov:<12} {mid:<36} {speed_val:>11.1f} t/s {src:<10} {m.get('output_limit',0):>10,}")

    def _prompt_custom_model(self) -> bool:
        print("\n=== Option 9: Configure Custom Model ===")
        print("Supported providers: ollama, gemini, groq, openrouter, nvidia, huggingface")
        raw = input("Enter custom model as 'provider:model_id' (or press Enter to choose provider separately, 'b' to go back): ").strip()
        if raw.lower() == "b":
            return False
        if ":" in raw and raw.split(":", 1)[0].lower() in PROVIDER_RATE_INFO:
            provider, model_id = raw.split(":", 1)
            provider, model_id = provider.strip().lower(), model_id.strip()
        else:
            provider = input("Enter provider (ollama/gemini/groq/openrouter/nvidia/huggingface): ").strip().lower()
            if provider not in PROVIDER_RATE_INFO:
                print(f"[Router] Invalid provider '{provider}'. Returning to menu.")
                return False
            model_id = raw if raw else input("Enter exact model ID: ").strip()
        if not model_id:
            print("[Router] Model ID cannot be empty. Returning to menu.")
            return False
        self.custom_model = {"provider": provider, "id": model_id}
        self.selected_option = "9"
        print(f"[Router] Active pipeline: Custom Model ({provider}:{model_id}) — will route back to choice menu on any error.")
        return True

    def select_provider_interactive(self, default_option: Optional[str] = None, vision_only: bool = False) -> str:
        if default_option in self.OPTIONS or default_option in ("7", "8"):
            self.selected_option = default_option
            return default_option
        while True:
            print("\n=== Select Free LLM API Pipeline ===")
            for key, label in self.OPTIONS.items():
                count = len(self._get_option_models(key, vision_only=vision_only))
                print(f"  {key}. {label} ({count} {'vision ' if vision_only else ''}models)")
            ranked_count = len(self._get_ranked_models(top_n=10, vision_only=vision_only))
            speed_count = len(self._get_speed_ranked_models(top_n=10, vision_only=vision_only))
            print("  6. Compare all models side-by-side by Token Limits (Top 10)")
            print(f"  7. Use Top-10 Token-Ranked Models (Rank 1 -> Rank {ranked_count} fallback)")
            print(f"  8. Compare & Rank Models by Output Speed (Top {speed_count} fastest tok/s fallback)")
            print("  9. Input a Custom Model (routes back to this menu if the model throws an error)")
            print("  r. Refresh free model lists from API keys")
            choice = input("Enter option (1-9 or r): ").strip().lower()
            if choice == "r":
                self.discover_and_cache_models()
                continue
            if choice == "6":
                self.compare_top_models(top_n=10, vision_only=vision_only)
                continue
            if choice == "7":
                if not ranked_count:
                    print("[Router] No ranked models available. Press 'r' to refresh.")
                    continue
                self.selected_option = "7"
                print(f"[Router] Active pipeline: Top-10 Token-Ranked Models (Rank 1 -> Rank {ranked_count})")
                return "7"
            if choice == "8":
                if not speed_count:
                    print("[Router] No models cached yet. Press 'r' to refresh.")
                    continue
                self.compare_models_by_speed(top_n=10, vision_only=vision_only)
                sub = input("\n[Option 8] Enter 'u' to use Speed-Ranked Top-10 pipeline, 'b' to run live speed benchmark, or Enter to return to menu: ").strip().lower()
                if sub == "b":
                    self.benchmark_models_speed(top_n=10, vision_only=vision_only)
                    self.compare_models_by_speed(top_n=10, vision_only=vision_only)
                    use_now = input("Use this Speed-Ranked Top-10 pipeline now? (y/n): ").strip().lower()
                    if use_now == "y":
                        self.selected_option = "8"
                        print(f"[Router] Active pipeline: Top-10 Speed-Ranked Models (Fastest 1 -> {speed_count} fallback)")
                        return "8"
                elif sub == "u":
                    self.selected_option = "8"
                    print(f"[Router] Active pipeline: Top-10 Speed-Ranked Models (Fastest 1 -> {speed_count} fallback)")
                    return "8"
                continue
            if choice == "9":
                if self._prompt_custom_model():
                    return "9"
                continue
            if choice in self.OPTIONS:
                if not self._get_option_models(choice, vision_only=vision_only):
                    print(f"[Router] No models cached for option {choice}. Press 'r' to refresh.")
                    continue
                self.selected_option = choice
                print(f"[Router] Active pipeline: {self.OPTIONS[choice]}")
                return choice
            print("Invalid selection. Enter 1-9 or r.")

    def query(self, prompt: str, system_prompt: Optional[str] = None, image_path: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None) -> tuple[str, str]:
        vision_only = bool(image_path)
        while True:
            if not self.selected_option:
                self.select_provider_interactive(vision_only=vision_only)

            # Option 9: Custom Model
            if self.selected_option == "9":
                if not self.custom_model:
                    self.selected_option = None
                    continue
                provider = self.custom_model["provider"]
                mid = self.custom_model["id"]
                print(f"[Router] Trying Custom Model {provider}:{mid} ...")
                try:
                    text = self._call_model(provider, mid, prompt, system_prompt, image_path, temperature, max_tokens)
                    return text, f"{provider}:{mid}"
                except Exception as exc:
                    print(f"\n[Custom Model Error] {provider}:{mid} threw an error: {exc}")
                    print("[Router] Routing you back to the choice options menu to select an alternative pipeline...")
                    self.selected_option = None
                    self.custom_model = None
                    continue

            # Other Options
            if self.selected_option == "7":
                candidates = self._get_ranked_models(top_n=10, vision_only=vision_only)
            elif self.selected_option == "8":
                candidates = self._get_speed_ranked_models(top_n=10, vision_only=vision_only)
            else:
                candidates = self._get_option_models(self.selected_option or "", vision_only=vision_only)

            if not candidates:
                print(f"[Router] No models available for option {self.selected_option}. Routing back to choice menu...")
                self.selected_option = None
                continue

            errors = []
            for idx, model in enumerate(candidates):
                mid, provider = model["id"], model["provider"]
                total = model.get("input_limit", 0) + model.get("output_limit", 0)
                tps = model.get("measured_tps") or model.get("est_tps") or _estimate_model_tps(provider, mid)
                print(f"[Router] Trying ({idx + 1}/{len(candidates)}) {provider}:{mid} [total_limit={total:,}, ~{float(tps):.0f} tok/s]")
                try:
                    text = self._call_model(provider, mid, prompt, system_prompt, image_path, temperature, max_tokens)
                    return text, f"{provider}:{mid}"
                except Exception as exc:
                    errors.append(f"{provider}:{mid} -> {exc}")
                    print(f"[Fallback] {provider}:{mid} failed ({exc}). Rerouting to next best model...")
                    
            print("[Router] All models in the selected option failed:\n" + "\n".join(errors))
            print("[Router] Routing back to the choice options menu...")
            self.selected_option = None

    def _call_model(self, provider: str, model_id: str, prompt: str, system_prompt: Optional[str], image_path: Optional[str], temperature: float, max_tokens: Optional[int]) -> str:
        b64_img, mime_type = _encode_image(image_path) if image_path else (None, None)

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
            if b64_img:
                user_msg["images"] = [b64_img]
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [user_msg]
            resp = requests.post(f"{base_url}/api/chat", json={"model": model_id, "messages": msgs, "stream": False}, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "")
            if not text: raise ValueError("Empty response from Ollama")
            return text

        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            parts: List[Dict[str, Any]] = []
            if b64_img: parts.append({"inline_data": {"mime_type": mime_type, "data": b64_img}})
            parts.append({"text": prompt})
            payload: Dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": temperature}}
            if system_prompt: payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            if max_tokens: payload["generationConfig"]["maxOutputTokens"] = max_tokens
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            out_parts = resp.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in out_parts)
            if not text: raise ValueError("Empty Gemini response")
            return text

        endpoints = {
            "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
            "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
            "huggingface": ("https://router.huggingface.co/v1/chat/completions", "HF_API_KEY"),
        }
        url, env_var = endpoints[provider]
        if b64_img:
            user_content: Any = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}]
        else:
            user_content = prompt
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": user_content}]
        payload = {"model": model_id, "messages": msgs, "temperature": temperature}
        if max_tokens: payload["max_tokens"] = max_tokens
        resp = requests.post(url, headers={"Authorization": f"Bearer {os.getenv(env_var)}", "Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        if not text: raise ValueError("Empty response")
        return text


# =====================================================================
# 3. TASK RUNNERS (Extract, Verify, Transcribe)
# =====================================================================

def task_extract(target: Path) -> None:
    if target.is_file():
        _extract_single(target)
    elif target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
        print(f"Found {len(pdfs)} PDFs in {target}")
        for pdf in pdfs:
            _extract_single(pdf)

def _extract_single(pdf_path: Path) -> None:
    doc = pymupdf.open(str(pdf_path))
    out_dir = Path("quant_chem") / "assets" / f"{pdf_path.stem}_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing '{pdf_path.name}' (Total Pages: {len(doc)})")
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=300)
        pix.save(str(out_dir / f"page_{page_num}.png"))
    print(f"  -> Rendered {len(doc)} pages to {out_dir}")

def task_verify(refresh: bool) -> None:
    router = FreeLLMRouter(force_refresh=refresh)
    router.select_provider_interactive(vision_only=True)
    test_img = Path("quant_chem/assets/Lecture_18_assets/page_1.png")
    if not test_img.exists():
        print(f"❌ Error: Could not find '{test_img}'. Run `python pipeline.py extract quant_chem/notes` first.")
        sys.exit(1)
    print(f"Testing Vision OCR on: {test_img}\n")
    try:
        answer, used_model = router.query(
            prompt="Read the lecture number and title from this slide in one short sentence.",
            image_path=str(test_img),
            max_tokens=100,
        )
        print(f"✅ {used_model}: {answer.strip()}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

def task_transcribe(target: Path, refresh: bool) -> None:
    SYSTEM_PROMPT = (
        "You are a quantum chemistry professor transcribing a lecture slide into clean Markdown. "
        "Use LaTeX ($ for inline, $$ for display) for ALL math variables, operators, wavefunctions, and equations. "
        "Preserve exact headings, tables, and bullet structure. "
        "For any chart, graph, or diagram, write a detailed explanation formatted as: "
        "> **Figure Description:** [detailed description of axes, curves, nodes, and physical meaning]. "
        "Output raw Markdown only without wrapping code fences."
    )
    USER_PROMPT = "Transcribe this quantum chemistry lecture page completely into clean Markdown + LaTeX."
    
    router = FreeLLMRouter(force_refresh=refresh)
    router.select_provider_interactive(vision_only=True)
    
    def _transcribe_folder(assets_dir: Path, output_md: Path):
        image_files = sorted([f for f in assets_dir.iterdir() if f.name.startswith("page_") and f.suffix == ".png"], key=lambda p: int(p.stem.split("_")[1]))
        if not image_files:
            return
        output_md.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Transcribing {assets_dir.name} ({len(image_files)} pages) -> {output_md} ===")
        with open(output_md, "w", encoding="utf-8") as f:
            for img_path in image_files:
                print(f"\nProcessing {img_path.name}...")
                try:
                    content, used_model = router.query(prompt=USER_PROMPT, system_prompt=SYSTEM_PROMPT, image_path=str(img_path), temperature=0.1)
                    f.write(f"<!-- Source: {img_path.name} | Model: {used_model} -->\n\n{content.strip()}\n\n---\n\n")
                    f.flush()
                    print(f"✅ Completed {img_path.name} via {used_model}")
                    time.sleep(1.5)
                except Exception as e:
                    print(f"❌ Failed on {img_path.name}: {e}")
                    f.write(f"> [!ERROR] Failed to transcribe {img_path.name}: {e}\n\n---\n\n")
                    f.flush()

    if target.is_dir() and any(target.glob("page_*.png")):
        lecture_name = target.name.replace("_assets", "")
        _transcribe_folder(target, Path("quant_chem/transcription") / f"{lecture_name}_notes.md")
    elif target.is_dir():
        for ldir in sorted([d for d in target.iterdir() if d.is_dir() and d.name.endswith("_assets")]):
            lecture_name = ldir.name.replace("_assets", "")
            out_path = Path("quant_chem/transcription") / f"{lecture_name}_notes.md"
            if out_path.exists() and out_path.stat().st_size > 200:
                print(f"[Skip] {out_path.name} already exists.")
                continue
            _transcribe_folder(ldir, out_path)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Chemistry Lecture Processing Pipeline")
    parser.add_argument("command", choices=["extract", "verify", "transcribe"], help="Task to run")
    parser.add_argument("target", nargs="?", default="", help="Target file or directory")
    parser.add_argument("--refresh-models", action="store_true", help="Force refresh cached API models")
    args = parser.parse_args()

    if args.command == "extract":
        target_path = Path(args.target) if args.target else Path("quant_chem/notes")
        task_extract(target_path)
    
    elif args.command == "verify":
        task_verify(args.refresh_models)
    
    elif arg_cmd := args.command == "transcribe":
        target_path = Path(args.target) if args.target else Path("quant_chem/assets")
        task_transcribe(target_path, args.refresh_models)
