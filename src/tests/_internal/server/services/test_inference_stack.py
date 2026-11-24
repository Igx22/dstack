import pytest

from dstack._internal.core.models.configurations import ServiceConfiguration
from dstack._internal.server.services.inference_stack import infer_inference_stack


def descriptor(**fields):
    if "groups" in fields:
        fields["groups"] = [{"replicas": 1, **group} for group in fields["groups"]]
    config = ServiceConfiguration.model_validate({"type": "service", "port": 8000, **fields})
    stack = infer_inference_stack(config)
    return stack.model_dump(mode="json") if stack else None


def routed(router, frontend, *workers):
    return descriptor(
        env={"MODEL_ID": "model"},
        groups=[
            {"router": {"type": router}, "commands": [frontend]},
            *({"commands": [command]} for command in workers),
        ],
    )


class TestInferInferenceStack:
    @pytest.mark.parametrize(("engine", "processor"), [("sglang", "vllm"), ("vllm", "dynamo")])
    def test_dynamo_frontend_and_homogeneous_workers(self, engine, processor):
        frontend = (
            'exec python3 -m dynamo.frontend \\\n  --http-port 8000 --dyn-chat-processor="%s"'
            % processor
        )
        worker = f"python3 -m dynamo.{engine} --model $MODEL_ID"
        assert descriptor(
            groups=[
                {
                    "name": "store",
                    "commands": ["curl installer | tar x", "etcd --data-dir /tmp/etcd"],
                },
                {
                    "router": {"type": "dynamo"},
                    "commands": ["export ETCD_ENDPOINTS=...", frontend],
                },
                {"commands": [worker + " --disaggregation-mode prefill"]},
                {"commands": [worker + " --disaggregation-mode decode"]},
            ]
        ) == {"engine": {"name": engine}, "router": {"name": "dynamo", "processor": processor}}

    @pytest.mark.parametrize(
        "frontend",
        [
            "python -m dynamo.frontend",
            "python -m dynamo.frontend --dyn-chat-processor unknown",
            "python -m dynamo.frontend --dyn-chat-processor $PROCESSOR",
            "python -m dynamo.frontend --dyn-chat-processor dynamo --dyn-chat-processor=vllm",
            "python -m dynamo.frontend --config config.yaml --dyn-chat-processor dynamo",
            "python -m dynamo.frontend --dyn-chat-processor dynamo $EXTRA_OPTS",
            "python -m dynamo.frontend --dyn-chat-processor dynamo --http-port $EXTRA_OPTS",
            "python -m dynamo.frontend -- --dyn-chat-processor dynamo",
            "python -m dynamo.frontend # --dyn-chat-processor dynamo",
            "echo 'python -m dynamo.frontend --dyn-chat-processor dynamo'",
            "python serve.py --dyn-chat-processor dynamo",
            "python -m dynamo.frontend --dyn-chat-processor dynamo | proxy",
            "python -m dynamo.frontend\necho --dyn-chat-processor dynamo",
            "python -m dynamo.frontend --dyn-chat-processor 'dynamo",
        ],
    )
    def test_dynamo_unresolved_processor(self, frontend):
        assert routed("dynamo", frontend, "python -m dynamo.vllm") == {
            "engine": {"name": "vllm"},
            "router": {"name": "dynamo"},
        }

    @pytest.mark.parametrize("other_worker", ["python serve.py", "python -m dynamo.sglang"])
    def test_unrecognized_or_mixed_workers(self, other_worker):
        assert routed(
            "dynamo",
            "python -m dynamo.frontend --dyn-chat-processor dynamo",
            "python -m dynamo.vllm",
            other_worker,
        ) == {
            "router": {"name": "dynamo", "processor": "dynamo"},
        }

    @pytest.mark.parametrize(
        ("worker", "engine", "transport"),
        [
            ("vllm serve $MODEL_ID --port 8000", "vllm", "http"),
            ("python -m vllm.entrypoints.openai.api_server", "vllm", "http"),
            ("python3 -m vllm.entrypoints.grpc_server", "vllm", "grpc"),
            ("python -m sglang.launch_server --model-path $MODEL_ID", "sglang", "http"),
            ("sglang serve --model-path test --grpc-mode", "sglang", "grpc"),
        ],
    )
    def test_smg_transport_comes_from_the_workers(self, worker, engine, transport):
        assert routed("smg", "smg launch --host 0.0.0.0", worker, worker) == {
            "engine": {"name": engine},
            "router": {"name": "smg", "transport": transport},
        }

    @pytest.mark.parametrize(
        ("frontend", "second"),
        [
            ("python custom_router.py", "vllm serve model"),
            ("smg launch", "python -m vllm.entrypoints.grpc_server"),
            ("smg launch", "vllm serve model --config config.yml"),
        ],
    )
    def test_smg_unknown_or_mixed_transport(self, frontend, second):
        assert routed("smg", frontend, "vllm serve model", second) == {
            "engine": {"name": "vllm"},
            "router": {"name": "smg"},
        }

    def test_direct_engine_in_shell_or_docker_argv(self):
        direct = {"engine": {"name": "vllm"}, "router": None}
        assert descriptor(commands=["vllm serve model"]) == direct
        assert descriptor(entrypoint="vllm", commands=["serve", "model"], image="custom") == direct

    @pytest.mark.parametrize(
        "earlier",
        [
            "python3 -m dynamo.frontend --dyn-chat-processor dynamo",
            "python3 -m dynamo.frontend --dyn-chat-processor dynamo &",
            "python launch_other_server.py",
        ],
    )
    def test_earlier_server_makes_group_launch_ambiguous(self, earlier):
        assert descriptor(
            groups=[
                {
                    "router": {"type": "dynamo"},
                    "commands": [earlier, "python3 -m dynamo.frontend --dyn-chat-processor vllm"],
                },
                {"commands": ["python3 -m dynamo.sglang"]},
            ]
        ) == {"engine": {"name": "sglang"}, "router": {"name": "dynamo"}}

    @pytest.mark.parametrize(
        "fields",
        [
            {"image": "vllm/vllm-openai:1"},
            {"image": "vllm/vllm-openai:1", "commands": ["python serve.py"]},
            {"commands": ["echo 'vllm serve model'"]},
            {"commands": ["vllm serve model &", "python wrapper.py"]},
            {"commands": ["vllm serve model"], "shell": "/bin/custom"},
            {"image": "custom", "entrypoint": "wrapper", "commands": ["vllm serve model"]},
        ],
    )
    def test_opaque_direct_launch(self, fields):
        assert descriptor(**fields) is None
