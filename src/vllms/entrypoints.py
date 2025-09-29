import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_served_model_name(model_path):
    if not model_path:
        return "unknown-model"
    model_name = model_path.split("/")[-1].lower()
    model_name = model_name.replace("-instruct", "-instruct")
    model_name = model_name.replace("-fp8", "")
    model_name = model_name.replace("-2507", "")
    return model_name


def setup_vllms():
    script_path = Path(__file__).parent.parent.parent / "scripts" / "setup_vllms.sh"
    if not script_path.exists():
        print(f"Error: Setup script not found at {script_path}")
        return 1
    print(f"Running setup script: {script_path}")
    return subprocess.call(["bash", str(script_path)])


def run_single(
    model_env, port_env, dtype_env, tool_parser="hermes", extra_flags: str = ""
):
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    model = os.getenv(model_env)
    served_name = get_served_model_name(model)
    port = os.getenv(port_env)
    dtype = os.getenv(dtype_env, "float16")

    extra = f"{extra_flags.strip()}" if extra_flags and extra_flags.strip() else ""
    
    # Build tool choice flags
    tool_flags = ""
    if "--disable-auto-tool-choice" not in extra:
        tool_flags = f"--enable-auto-tool-choice --tool-call-parser {tool_parser}"

    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            f"vllm-{served_name},caddy",
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model} "
            f"--served-model-name {served_name} "
            f"--dtype {dtype} "
            f"--host 0.0.0.0 --port {port} "
            f"{tool_flags} {extra}",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )


def run_qwen():
    qwen_flags = os.getenv("QWEN_FLAGS", "")
    return run_single(
        "QWEN_MODEL",
        "QWEN_PORT",
        "DTYPE_QWEN",
        tool_parser="hermes",
        extra_flags=qwen_flags,
    )


def run_deepseek():
    return run_single(
        "DEEPSEEK_MODEL", "DEEPSEEK_PORT", "DTYPE_DEEPSEEK", tool_parser="hermes"
    )


def run_mistral():
    mistral_flags = os.getenv("MISTRAL_FLAGS", "")
    return run_single(
        "MISTRAL_MODEL",
        "MISTRAL_PORT",
        "DTYPE_MISTRAL",
        tool_parser="mistral",
        extra_flags=mistral_flags,
    )


def run_llama3():
    llama3_flags = os.getenv("LLAMA3_FLAGS", "--enable_auto_tool_choice --tool-call-parser llama3_json --chat-template src/vllms/examples/tool_chat_template_llama3.1_json.jinja")
    return run_single(
        "LLAMA3_MODEL",
        "LLAMA3_PORT", 
        "DTYPE_LLAMA3",
        extra_flags=llama3_flags,
    )


def run_llama4_scout():
    llama4_flags = os.getenv("LLAMA4_SCOUT_FLAGS", "--quantization fp8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --chat-template examples/tool_chat_template_llama4_pythonic.jinja --override-generation-config='{\"attn_temperature_tuning\": true}'")
    return run_single(
        "LLAMA4_SCOUT_MODEL",
        "LLAMA4_SCOUT_PORT",
        "DTYPE_LLAMA4_SCOUT", 
        tool_parser="pythonic",
        extra_flags=llama4_flags,
    )
