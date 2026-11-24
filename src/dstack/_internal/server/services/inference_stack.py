"""Describe recognized inference launches from the stored service configuration."""

import re
import shlex

from dstack._internal.core.models.configurations import ReplicaGroup, ServiceConfiguration
from dstack._internal.core.models.inference_stack import (
    DYNAMO_NATIVE_PROCESSOR,
    InferenceEngine,
    InferenceRouter,
    InferenceStack,
)
from dstack._internal.core.models.routers import RouterType


def infer_inference_stack(configuration: ServiceConfiguration) -> InferenceStack | None:
    groups = configuration.replica_groups
    router_group = next((group for group in groups if group.router is not None), None)
    router_type = router_group.router.type if router_group and router_group.router else None
    workers: set[tuple[str | None, str | None]] = set()
    for group in groups:
        if group is router_group:
            continue
        args = _launch_args(configuration, group)
        # Dynamo examples have a dedicated etcd group for service discovery.
        if router_type == RouterType.DYNAMO and args[:1] == ["etcd"]:
            continue
        workers.add(_worker(args, router_type))

    engines = {engine for engine, _ in workers}
    engine_name = next(iter(engines)) if len(engines) == 1 else None
    engine = InferenceEngine(name=engine_name) if engine_name else None
    if router_group is None or router_type is None:
        # A homogeneous set of directly launched HTTP/gRPC servers has no frontend.
        return InferenceStack(engine=engine, router=None) if engine else None

    frontend = _launch_args(configuration, router_group)
    processor = None
    transport = None
    if router_type == RouterType.DYNAMO and frontend[:3] == ["python", "-m", "dynamo.frontend"]:
        processor = _dynamo_processor(frontend[3:])
    if router_type == RouterType.SMG and (
        frontend[:2] == ["smg", "launch"]
        or frontend[:3] == ["python", "-m", "sglang_router.launch_router"]
    ):
        transports = {protocol for _, protocol in workers}
        if len(transports) == 1:
            transport = next(iter(transports))
    return InferenceStack(
        engine=engine,
        router=InferenceRouter(name=router_type.value, processor=processor, transport=transport),
    )


def _launch_args(configuration: ServiceConfiguration, group: ReplicaGroup) -> list[str]:
    # Match the job configurator: an explicit Docker entrypoint receives commands as
    # argv. Otherwise commands are shell snippets, with setup preceding the final launch.
    try:
        if configuration.entrypoint is not None:
            args = shlex.split(configuration.entrypoint) + configuration.commands
        elif group.commands and configuration.shell in (
            None,
            "sh",
            "/bin/sh",
            "bash",
            "/bin/bash",
        ):
            if not all(_is_setup_command(command) for command in group.commands[:-1]):
                return []
            args = _shell_args(group.commands[-1])
            # An unquoted variable can expand into additional CLI flags. Mask only
            # known single-word values; metadata always comes from literal arguments.
            scalar_names = {"DSTACK_GPUS_NUM"} | {
                name
                for name, value in configuration.env.items()
                if isinstance(value, str) and re.fullmatch(r"[\w./:][\w./:,-]*", value)
            }
            scalar_args = {
                "$" + name
                for name in scalar_names
                if not any(name + "=" in command for command in group.commands[:-1])
            }
            args = ["<value>" if arg in scalar_args else arg for arg in args]
            if args[:1] == ["exec"]:
                args = args[1:]
        else:
            return []  # The image entrypoint is unknown.
    except ValueError:
        return []
    if args[:1] == ["python3"]:
        args[0] = "python"
    return args


def _is_setup_command(command: str) -> bool:
    # An earlier serving process can block the last command or continue in the background.
    # Recognize the setup steps in the supported launch recipes.
    if "|" in command:
        download, _, unpack = command.replace("\\\n", "").partition("|")
        return _shell_args(download)[:1] == ["curl"] and _shell_args(unpack)[:1] == ["tar"]
    args = _shell_args(command)
    return args[:1] in (["export"], ["echo"], ["cd"]) or args[:2] == ["pip", "install"]


def _shell_args(command: str) -> list[str]:
    command = command.strip().replace("\\\n", "")
    if any(char in command for char in "\n;&|<>`#") or "$(" in command:
        return []
    return shlex.split(command)


def _worker(args: list[str], router_type: RouterType | None) -> tuple[str | None, str | None]:
    if router_type == RouterType.DYNAMO:
        if args[:3] == ["python", "-m", "dynamo.vllm"]:
            return "vllm", None
        if args[:3] == ["python", "-m", "dynamo.sglang"]:
            return "sglang", None
        return None, None
    if args[:3] == ["python", "-m", "vllm.entrypoints.grpc_server"]:
        return "vllm", "grpc"
    if args[:2] == ["vllm", "serve"] or args[:3] == [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]:
        if _has_indirect_options(args) or any(arg.startswith("--grpc") for arg in args):
            return "vllm", None
        return "vllm", "http"
    if args[:2] == ["sglang", "serve"] or args[:3] == ["python", "-m", "sglang.launch_server"]:
        if "--grpc-mode" in args:
            return "sglang", "grpc"
        if _has_indirect_options(args) or any(arg.startswith("--grpc-mode=") for arg in args):
            return "sglang", None
        return "sglang", "http"
    return None, None


def _dynamo_processor(args: list[str]) -> str | None:
    if _has_indirect_options(args):
        return None
    values = set()
    for index, arg in enumerate(args):
        if arg == "--dyn-chat-processor":
            values.add(args[index + 1] if index + 1 < len(args) else None)
        elif arg.startswith("--dyn-chat-processor="):
            values.add(arg.partition("=")[2])
    if len(values) == 1:
        value = values.pop()
        if value in (DYNAMO_NATIVE_PROCESSOR, "vllm"):
            return value
    return None


def _has_indirect_options(args: list[str]) -> bool:
    return any(
        "$" in arg or arg.partition("=")[0] in ("--", "--config", "--config-file") for arg in args
    )
