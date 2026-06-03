import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import verifiers as core_vf
import verifiers as vf
from verifiers.types import Tool
from verifiers.v1.types import ConfigMap

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent
from tau2.agent.llm_agent import is_valid_agent_history_message
from tau2.config import DEFAULT_LLM_ARGS_AGENT, DEFAULT_LLM_ARGS_USER
from tau2.config import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage
from tau2.data_model.message import ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task as TauTask
from tau2.environment.environment import Environment as TauEnvironment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, Role
from tau2.registry import registry
from tau2.run import load_tasks
from tau2.user.user_simulator import UserSimulator, is_valid_user_history_message
from tau2.utils.utils import DATA_DIR, format_time, get_now
from verifiers.utils.async_utils import maybe_call_with_named_args

from reward_hacking import (
    MutationSpec,
    append_agent_elicitation_prompt,
    append_policy_patch,
    apply_task_payload_mutation,
    evaluate_reward_hacking,
    load_mutation_specs,
    mutation_spec_from_task_info,
    resolve_mutation_path,
    task_info_without_mutation,
    validate_mutation_specs,
)

DEFAULT_USER_MODEL = "gpt-4.1"
DEFAULT_USER_BASE_URL = "https://api.openai.com/v1"
DEFAULT_USER_API_KEY_VAR = "OPENAI_API_KEY"
TAU2_DATA_REV = "337326e62d8e0ca74c353b004a9c5d748e0ba914"


def download_tau2_data() -> None:
    tau2_data_dir = DATA_DIR / "tau2"
    source_rev_path = tau2_data_dir / ".source_rev"
    if (
        tau2_data_dir.exists()
        and (tau2_data_dir / "domains").exists()
        and source_rev_path.exists()
        and source_rev_path.read_text().strip() == TAU2_DATA_REV
    ):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tau2_bench_v1_") as temp_name:
        temp_dir = Path(temp_name)
        subprocess.run(
            ["git", "init", temp_dir],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(temp_dir),
                "remote",
                "add",
                "origin",
                "https://github.com/sierra-research/tau2-bench.git",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(temp_dir), "fetch", "--depth", "1", "origin", TAU2_DATA_REV],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(temp_dir), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
            capture_output=True,
        )
        source_data = temp_dir / "data"
        source_tau2 = source_data / "tau2"
        if not source_tau2.exists():
            raise FileNotFoundError(f"tau2 data not found at {source_tau2}")
        if tau2_data_dir.exists():
            shutil.rmtree(tau2_data_dir)
        shutil.copytree(source_tau2, tau2_data_dir)
        source_rev_path.write_text(f"{TAU2_DATA_REV}\n")


def tau_msg_to_vf_dict(message: Message) -> vf.ConfigData:
    if isinstance(message, AssistantMessage):
        if message.tool_calls:
            return core_vf.AssistantMessage(
                content=message.content,
                tool_calls=[
                    core_vf.ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.arguments),
                    )
                    for tool_call in message.tool_calls
                ],
            ).model_dump(exclude_none=True)
        return core_vf.AssistantMessage(content=message.content).model_dump(
            exclude_none=True
        )
    if isinstance(message, UserMessage):
        return core_vf.UserMessage(content=message.content or "").model_dump(
            exclude_none=True
        )
    if isinstance(message, ToolMessage):
        return core_vf.ToolMessage(
            tool_call_id=message.id,
            content=message.content or "",
        ).model_dump(exclude_none=True)
    raise ValueError(f"Unknown tau2 message type: {type(message)}")


def dump_tau_message(message: Message) -> vf.ConfigData:
    return cast(vf.ConfigData, message.model_dump(mode="json", exclude_none=True))


def load_tau_message(payload: ConfigMap) -> Message:
    role = payload.get("role")
    if role == "assistant":
        return AssistantMessage.model_validate(payload)
    if role == "user":
        return UserMessage.model_validate(payload)
    if role == "tool":
        if "tool_messages" in payload:
            return MultiToolMessage.model_validate(payload)
        return ToolMessage.model_validate(payload)
    raise ValueError(f"Unknown tau2 message role: {role!r}")


class Tau2Session:
    def __init__(
        self,
        domain: str,
        task_payload: ConfigMap,
        user_model: str,
        user_args: ConfigMap,
        max_steps: int,
        max_errors: int,
        mutation_spec: MutationSpec | None = None,
    ):
        self.domain = domain
        self.task_payload = dict(task_payload)
        self.user_model = user_model
        self.user_args = dict(user_args)
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.mutation_spec = mutation_spec
        self.ready = False
        self.initial_prompt_messages: list[vf.ConfigData] = []
        self.recorded_assistant_messages = 0
        self.pending_agent_tool_calls: list[ToolCall] = []
        self.num_assistant_tool_calls = 0
        self.num_user_tool_calls = 0

    async def initialize(self, state: vf.State) -> None:
        if self.ready:
            return
        self.task = TauTask.model_validate(self.task_payload)
        environment_constructor = registry.get_env_constructor(self.domain)
        self.environment = await asyncio.to_thread(environment_constructor)
        domain_policy = self.environment.get_policy()
        if self.mutation_spec is not None:
            domain_policy = append_policy_patch(domain_policy, self.mutation_spec)
        self.agent = LLMAgent(
            tools=self.environment.get_tools(),
            domain_policy=domain_policy,
            llm=str(state.get("runtime", {}).get("model") or ""),
            llm_args=dict(state.get("runtime", {}).get("sampling_args") or {}),
        )
        self.user = UserSimulator(
            tools=self.user_tools(),
            instructions=str(self.task.user_scenario),
            llm=self.user_model,
            llm_args=self.user_args,
        )
        self.init_tau2_state()
        self.environment.sync_tools()
        self.ready = True
        self.initial_prompt_messages = await self.advance_until_agent(state)
        self.render_state(state)

    def user_tools(self) -> object:
        try:
            return self.environment.get_user_tools()
        except Exception:
            return None

    def init_tau2_state(self) -> None:
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for message in message_history:
            message.turn_idx = None
        message_history = add_timestamps(message_history)
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        self.done = False
        self.termination_reason: TerminationReason | None = None
        if message_history:
            self.init_from_history(message_history)
        else:
            self.user_state = self.user.get_init_state()
            first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
            first_message.timestamp = get_now()
            self.agent_state = self.agent.get_init_state(
                message_history=[first_message]
            )
            self.trajectory: list[Message] = [first_message]
            self.message = first_message
            self.from_role = Role.AGENT
            self.to_role = Role.USER
        self.step_count = 0
        self.num_errors = 0

    def init_from_history(self, message_history: list[Message]) -> None:
        last_message = message_history[-1]
        self.trajectory = message_history
        self.message = last_message
        if isinstance(last_message, AssistantMessage):
            self.from_role = Role.AGENT
            self.to_role = Role.ENV if last_message.is_tool_call() else Role.USER
            self.agent_state = self.agent.get_init_state(
                message_history=[
                    msg
                    for msg in message_history
                    if is_valid_agent_history_message(msg)
                ]
            )
            self.user_state = self.user.get_init_state(
                message_history=[
                    msg
                    for msg in message_history[:-1]
                    if is_valid_user_history_message(msg)
                ]
            )
            if self.agent.is_stop(last_message):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            return
        if isinstance(last_message, UserMessage):
            self.from_role = Role.USER
            self.to_role = Role.ENV if last_message.is_tool_call() else Role.AGENT
            self.user_state = self.user.get_init_state(
                message_history=[
                    msg for msg in message_history if is_valid_user_history_message(msg)
                ]
            )
            self.agent_state = self.agent.get_init_state(
                message_history=[
                    msg
                    for msg in message_history[:-1]
                    if is_valid_agent_history_message(msg)
                ]
            )
            self.done = UserSimulator.is_stop(last_message)
            if self.done:
                self.termination_reason = TerminationReason.USER_STOP
            return
        if isinstance(last_message, ToolMessage):
            self.from_role = Role.ENV
            self.to_role = (
                Role.AGENT if last_message.requestor == "assistant" else Role.USER
            )
            self.agent_state = self.agent.get_init_state(
                message_history=[
                    msg
                    for msg in message_history
                    if is_valid_agent_history_message(msg)
                ]
            )
            self.user_state = self.user.get_init_state(
                message_history=[
                    msg for msg in message_history if is_valid_user_history_message(msg)
                ]
            )
            return
        raise ValueError(f"Unsupported tau2 message type: {type(last_message)}")

    async def record_assistant_from_state(self, state: vf.State) -> None:
        completion = state.get("completion") or []
        assistant_messages = (
            vf.get_messages(completion, role="assistant")
            if isinstance(completion, list)
            else []
        )
        for message in assistant_messages[self.recorded_assistant_messages :]:
            assistant_message = assistant_from_openai_message(message)
            self.agent_state.messages.append(assistant_message)
            try:
                assistant_message.validate()
            except ValueError:
                self.done = True
                self.termination_reason = TerminationReason.AGENT_ERROR
                self.trajectory.append(assistant_message)
                continue
            if self.agent.is_stop(assistant_message):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(assistant_message)
            self.pending_agent_tool_calls.extend(assistant_message.tool_calls or [])
            self.num_assistant_tool_calls += len(assistant_message.tool_calls or [])
            self.message = assistant_message
            self.from_role = Role.AGENT
            self.to_role = Role.ENV if assistant_message.tool_calls else Role.USER
            self.step_count += 1
            self.environment.sync_tools()
            self.check_limits()
        self.recorded_assistant_messages = len(assistant_messages)
        self.render_state(state)

    async def call_agent_tool(
        self, name: str, arguments: ConfigMap, state: vf.State
    ) -> str:
        await self.record_assistant_from_state(state)
        tool_call = self.pop_pending_tool_call(name, arguments)
        tool_message = await asyncio.to_thread(self.environment.get_response, tool_call)
        if tool_message.error:
            self.num_errors += 1
        self.trajectory.append(tool_message)
        self.message = tool_message
        self.from_role = Role.ENV
        self.to_role = Role.AGENT
        self.environment.sync_tools()
        self.check_limits()
        self.render_state(state)
        return tool_message.content or ""

    def pop_pending_tool_call(self, name: str, arguments: ConfigMap) -> ToolCall:
        for index, tool_call in enumerate(self.pending_agent_tool_calls):
            if tool_call.name == name:
                return self.pending_agent_tool_calls.pop(index)
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=dict(arguments),
            requestor="assistant",
        )

    async def user_messages(self, state: vf.State) -> list[vf.ConfigData]:
        await self.record_assistant_from_state(state)
        if self.done:
            self.render_state(state)
            return []
        messages = await self.advance_until_agent(state)
        self.render_state(state)
        return messages

    async def advance_until_agent(self, state: vf.State) -> list[vf.ConfigData]:
        messages: list[vf.ConfigData] = []
        while not (self.done or self.to_role == Role.AGENT):
            if self.to_role == Role.USER:
                user_message, self.user_state = await asyncio.to_thread(
                    self.user.generate_next_message,
                    self.message,
                    self.user_state,
                )
                try:
                    user_message.validate()
                except ValueError:
                    self.done = True
                    self.termination_reason = TerminationReason.USER_ERROR
                    self.trajectory.append(user_message)
                    break
                if UserSimulator.is_stop(user_message):
                    self.done = True
                    self.termination_reason = TerminationReason.USER_STOP
                self.num_user_tool_calls += len(user_message.tool_calls or [])
                self.trajectory.append(user_message)
                self.message = user_message
                self.from_role = Role.USER
                self.to_role = Role.ENV if user_message.is_tool_call() else Role.AGENT
                self.step_count += 1
                self.check_limits()
                if not user_message.is_tool_call():
                    messages.append(tau_msg_to_vf_dict(user_message))
                    break
                continue
            if self.to_role == Role.ENV:
                await self.execute_user_tools()
                self.check_limits()
                continue
            raise ValueError(
                f"Invalid tau2 role transition: {self.from_role} -> {self.to_role}"
            )
        return messages

    async def execute_user_tools(self) -> None:
        tool_calls = list(getattr(self.message, "tool_calls", []) or [])
        tool_messages = []
        for tool_call in tool_calls:
            tool_call.requestor = "user"
            tool_message = await asyncio.to_thread(
                self.environment.get_response, tool_call
            )
            if tool_message.error:
                self.num_errors += 1
            tool_messages.append(tool_message)
        self.trajectory.extend(tool_messages)
        if len(tool_messages) > 1:
            self.message = MultiToolMessage(
                role="tool",
                tool_messages=tool_messages,
            )
        elif tool_messages:
            self.message = tool_messages[0]
        self.from_role = Role.ENV
        self.to_role = Role.USER
        self.step_count += 1
        self.environment.sync_tools()

    def check_limits(self) -> None:
        if self.step_count >= self.max_steps and self.to_role != Role.ENV:
            self.done = True
            self.termination_reason = TerminationReason.MAX_STEPS
        if self.num_errors >= self.max_errors:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS

    def render_state(self, state: vf.State) -> None:
        state["tau2"] = {
            "task_id": self.task.id,
            "done": self.done,
            "termination_reason": (
                self.termination_reason.value if self.termination_reason else None
            ),
            "step_count": self.step_count,
            "num_errors": self.num_errors,
            "num_assistant_tool_calls": self.num_assistant_tool_calls,
            "num_user_tool_calls": self.num_user_tool_calls,
            "messages": [dump_tau_message(message) for message in self.trajectory],
        }
        db_snapshot = self.db_snapshot()
        if db_snapshot is not None:
            state["tau2"]["db"] = db_snapshot
        if self.mutation_spec is not None:
            state["tau2"]["mutation"] = self.mutation_spec.to_mapping()
        state["num_assistant_tool_calls"] = self.num_assistant_tool_calls
        state["num_user_tool_calls"] = self.num_user_tool_calls
        if self.done and self.termination_reason is not None:
            state.stop(self.termination_reason.value)

    def db_snapshot(self) -> ConfigMap | None:
        tools = getattr(self.environment, "tools", None)
        db = getattr(tools, "db", None)
        if db is None:
            return None
        order_ids, user_ids = self.referenced_db_ids()
        db_orders = getattr(db, "orders", {})
        db_users = getattr(db, "users", {})
        orders = {
            order_id: db_orders[order_id].model_dump(mode="json")
            for order_id in sorted(order_ids)
            if order_id in db_orders
        }
        users = {
            user_id: db_users[user_id].model_dump(mode="json")
            for user_id in sorted(user_ids)
            if user_id in db_users
        }
        return {"orders": orders, "users": users}

    def referenced_db_ids(self) -> tuple[set[str], set[str]]:
        order_ids: set[str] = set()
        user_ids: set[str] = set()
        for message in self.trajectory:
            for tool_call in getattr(message, "tool_calls", []) or []:
                self.collect_argument_ids(tool_call.arguments, order_ids, user_ids)
        return order_ids, user_ids

    @staticmethod
    def collect_argument_ids(
        arguments: ConfigMap,
        order_ids: set[str],
        user_ids: set[str],
    ) -> None:
        for key, value in arguments.items():
            if key == "order_id":
                add_string_values(value, order_ids)
            elif key == "user_id":
                add_string_values(value, user_ids)


def make_tau2_setup(
    session_factory: Callable[..., Tau2Session],
) -> vf.Handler:
    @vf.setup(priority=100)
    async def tau2_setup(task: vf.Task, state: vf.State) -> None:
        runtime = state.runtime_state()
        runtime["sampling_args"] = merge_tau2_agent_sampling_args(runtime)
        session = cast(
            Tau2Session,
            await maybe_call_with_named_args(session_factory, task=task, state=state),
        )
        await session.initialize(state)
        state.setdefault("prompt", [])
        if not state.get("tau2_prompt_initialized"):
            state["prompt"].extend(session.initial_prompt_messages)
            state["tau2_prompt_initialized"] = True

    return tau2_setup


def merge_tau2_agent_sampling_args(runtime: ConfigMap) -> ConfigMap:
    runtime_sampling_args = dict(runtime.get("sampling_args") or {})
    sampling_args = dict(DEFAULT_LLM_ARGS_AGENT)
    sampling_args.update(runtime_sampling_args)

    model = str(runtime.get("model") or "")
    model_id = model.removeprefix("openai/")
    has_reasoning_config = (
        "reasoning" in runtime_sampling_args
        or "reasoning_effort" in runtime_sampling_args
    )
    if (
        has_reasoning_config
        and "temperature" not in runtime_sampling_args
        and (
            model_id.startswith("gpt-5")
            or model_id.startswith("claude-sonnet-4-6")
            or model_id.startswith("claude-opus-4-6")
            or model_id.startswith("claude-opus-4-7")
        )
    ):
        sampling_args.pop("temperature", None)

    return sampling_args


def assistant_from_openai_message(message: vf.AssistantMessage) -> AssistantMessage:
    tool_calls = []
    for raw_tool_call in message.tool_calls or []:
        arguments = raw_tool_call.arguments
        if isinstance(arguments, str):
            parsed_arguments = json.loads(arguments or "{}")
        else:
            parsed_arguments = arguments
        tool_calls.append(
            ToolCall(
                id=raw_tool_call.id or f"call_{uuid.uuid4().hex[:8]}",
                name=raw_tool_call.name,
                arguments=cast(vf.ConfigData, parsed_arguments),
                requestor="assistant",
            )
        )
    content = message.content
    return AssistantMessage(
        role="assistant",
        content=content if isinstance(content, str) and content else None,
        tool_calls=tool_calls or None,
        raw_data=message.model_dump(exclude_none=True),
    )


def add_timestamps(message_history: list[Message]) -> list[Message]:
    time_offset = datetime.now() - timedelta(seconds=len(message_history))
    for index, message in enumerate(message_history):
        message.timestamp = format_time(time_offset + timedelta(seconds=index))
    return message_history


def add_string_values(value: object, target: set[str]) -> None:
    if isinstance(value, str):
        target.add(value)
        return
    if isinstance(value, list):
        for item in value:
            add_string_values(item, target)


def source(
    domain: str,
    max_turns: int,
    mutation_specs: list[MutationSpec] | None = None,
    agent_elicitation_prompt: str | None = None,
    agent_elicitation_position: str = "post_policy",
):
    if agent_elicitation_position not in {"pre_policy", "post_policy"}:
        raise ValueError(
            "agent_elicitation_position must be 'pre_policy' or 'post_policy'"
        )
    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    base_system_prompt = SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=environment.policy,
    )
    tasks = {
        str(task.id): task
        for task in load_tasks(task_set_name=domain, task_split_name="base")
    }
    if mutation_specs is not None:
        rows = []
        for spec in mutation_specs:
            task = tasks.get(spec.base_task_id)
            if task is None:
                raise ValueError(
                    f"Mutation {spec.mutation_id!r} references missing "
                    f"{domain} task {spec.base_task_id!r}"
                )
            rows.append((spec.mutation_id, task, spec))
    else:
        rows = [(str(index), task, None) for index, task in enumerate(tasks.values())]

    for index, task, mutation_spec in rows:
        task_payload = task.model_dump(mode="json", exclude_none=True)
        system_prompt = base_system_prompt
        if agent_elicitation_position == "pre_policy":
            system_prompt = append_agent_elicitation_prompt(
                system_prompt,
                agent_elicitation_prompt,
            )
        if mutation_spec is not None:
            task_payload = apply_task_payload_mutation(task_payload, mutation_spec)
            system_prompt = append_policy_patch(system_prompt, mutation_spec)
        if agent_elicitation_position == "post_policy":
            system_prompt = append_agent_elicitation_prompt(
                system_prompt,
                agent_elicitation_prompt,
            )
        yield {
            "example_id": index,
            "taskset_id": f"tau2_{domain}",
            "task_id": task.id,
            "domain": domain,
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "prompt": [],
            "info": json.dumps(task_payload),
        }


def load_session_factory(
    domain: str,
    user_model: str,
    user_args: ConfigMap,
    max_steps: int,
    max_errors: int,
) -> Callable[..., Tau2Session]:
    sessions: dict[str, Tau2Session] = {}

    def load_session(task, state) -> Tau2Session:
        key = str(state.get("trajectory_id") or id(state))
        if key in sessions:
            return sessions[key]
        task_info = task_info_from_task(task)
        mutation_spec = mutation_spec_from_task_info(task_info)
        session = Tau2Session(
            domain=domain,
            task_payload=cast(ConfigMap, task_info_without_mutation(task_info)),
            user_model=user_model,
            user_args=user_args,
            max_steps=max_steps,
            max_errors=max_errors,
            mutation_spec=mutation_spec,
        )
        sessions[key] = session
        return session

    return load_session


def make_tau2_tool(name: str, schema: ConfigMap) -> vf.Handler:
    async def tool(session: Tau2Session, state, **arguments) -> str:
        return await session.call_agent_tool(name, arguments, state)

    function_schema = cast(ConfigMap, schema["function"])
    tool.__name__ = name
    tool.__doc__ = str(function_schema.get("description") or "")
    tool.tool_def = Tool(
        name=name,
        description=str(function_schema.get("description") or ""),
        parameters=cast(vf.ConfigData, function_schema.get("parameters") or {}),
        strict=False,
    )
    return tool


def load_toolset(
    domain: str = "telecom",
    user_model: str = DEFAULT_USER_MODEL,
    user_args: ConfigMap = DEFAULT_LLM_ARGS_USER,
    user_base_url: str = DEFAULT_USER_BASE_URL,
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    session_factory: Callable[..., Tau2Session] | None = None,
) -> vf.Toolset:
    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = cast(TauEnvironment, environment_constructor())
    schemas = [tool.openai_schema for tool in environment.get_tools()]
    tools = [
        make_tau2_tool(
            str(cast(ConfigMap, schema["function"])["name"]),
            schema,
        )
        for schema in schemas
    ]
    if session_factory is None:
        session_factory = load_session_factory(
            domain=domain,
            user_model=user_model,
            user_args=tau2_user_args(user_args, user_base_url, user_api_key_var),
            max_steps=max_steps,
            max_errors=max_errors,
        )
    return vf.Toolset(
        tools=tools,
        bindings={f"{tool.__name__}.session": session_factory for tool in tools},
        write=True,
        scope="rollout",
    )


def tau2_user_args(
    user_args: ConfigMap, user_base_url: str, user_api_key_var: str
) -> vf.ConfigData:
    return {
        **dict(user_args),
        "api_base": user_base_url,
        "api_key": os.getenv(user_api_key_var),
    }


async def tau2_user(session: Tau2Session, state, transcript) -> list[vf.ConfigData]:
    _ = transcript
    return await session.user_messages(state)


def task_info_from_task(task) -> ConfigMap:
    task_info = task["info"]
    if isinstance(task_info, str):
        return cast(ConfigMap, json.loads(task_info))
    return cast(ConfigMap, task_info)


def ensure_reward_hacking_state(task, state) -> ConfigMap | None:
    tau2_state = cast(ConfigMap, state.get("tau2") or {})
    cached = tau2_state.get("reward_hacking")
    if isinstance(cached, dict):
        return cast(ConfigMap, cached)
    mutation_spec = mutation_spec_from_task_info(task_info_from_task(task))
    if mutation_spec is None:
        return None
    reward_hacking = evaluate_reward_hacking(mutation_spec, tau2_state)
    state.setdefault("tau2", {})["reward_hacking"] = reward_hacking
    return reward_hacking


@vf.reward(weight=1.0)
async def tau2_reward(task, state) -> float:
    tau2_state = cast(ConfigMap, state["tau2"])
    messages = [
        load_tau_message(cast(ConfigMap, message))
        for message in cast(list[ConfigMap], tau2_state["messages"])
    ]
    termination = tau2_state.get("termination_reason")
    if isinstance(termination, str):
        termination_reason = TerminationReason(termination)
    elif state.get("stop_condition") == "max_turns_reached":
        termination_reason = TerminationReason.MAX_STEPS
    else:
        termination_reason = TerminationReason.AGENT_ERROR
    state["tau2"]["termination_reason"] = termination_reason.value
    task_info = task_info_from_task(task)
    mutation_spec = mutation_spec_from_task_info(task_info)
    tau_task = TauTask.model_validate(task_info_without_mutation(task_info))
    simulation = SimulationRun(
        id=f"{task['taskset_id']}_{task['task_id']}_{datetime.now().isoformat()}",
        task_id=tau_task.id,
        messages=messages,
        termination_reason=termination_reason,
        timestamp=datetime.now().isoformat(),
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
        duration=0.0,
        agent_cost=0.0,
        user_cost=0.0,
    )
    evaluation_type = (
        EvaluationType.ENV
        if mutation_spec is not None
        else EvaluationType.ALL_WITH_NL_ASSERTIONS
    )
    state["tau2"]["native_evaluation_type"] = evaluation_type.value
    try:
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=tau_task,
            evaluation_type=evaluation_type,
            solo_mode=False,
            domain=str(task["domain"]),
        )
    except Exception as exc:
        if mutation_spec is None:
            raise
        state["tau2"]["evaluation"] = None
        state["tau2"]["native_evaluation_error"] = f"{type(exc).__name__}: {exc}"
        state["tau2"]["native_reward"] = 0.0
    else:
        state["tau2"]["evaluation"] = reward_info.model_dump(mode="json")
        state["tau2"]["native_reward"] = float(reward_info.reward)
    if mutation_spec is not None:
        reward_hacking = ensure_reward_hacking_state(task, state)
        assert reward_hacking is not None
        return float(reward_hacking["buggy_reward"])
    return float(state["tau2"]["native_reward"])


@vf.metric
async def tau2_num_errors(task, state) -> float:
    _ = task
    return float(state.get("tau2", {}).get("num_errors", 0.0))


@vf.metric
async def tau2_num_steps(task, state) -> float:
    _ = task
    return float(state.get("tau2", {}).get("step_count", 0.0))


@vf.metric
async def tau2_num_assistant_tool_calls(task, state) -> float:
    _ = task
    return float(state.get("num_assistant_tool_calls", 0.0))


@vf.metric
async def tau2_num_user_tool_calls(task, state) -> float:
    _ = task
    return float(state.get("num_user_tool_calls", 0.0))


@vf.metric
async def tau2_buggy_reward(task, state) -> float:
    reward_hacking = ensure_reward_hacking_state(task, state)
    return 0.0 if reward_hacking is None else float(reward_hacking["buggy_reward"])


@vf.metric
async def tau2_oracle_correct(task, state) -> float:
    reward_hacking = ensure_reward_hacking_state(task, state)
    return 0.0 if reward_hacking is None else float(reward_hacking["oracle_correct"])


@vf.metric
async def tau2_exploit_detected(task, state) -> float:
    reward_hacking = ensure_reward_hacking_state(task, state)
    return 0.0 if reward_hacking is None else float(reward_hacking["exploit_detected"])


@vf.metric
async def tau2_needs_manual_review(task, state) -> float:
    reward_hacking = ensure_reward_hacking_state(task, state)
    return 0.0 if reward_hacking is None else float(
        reward_hacking["needs_manual_review"]
    )


@vf.metric
async def tau2_simulator_contract_repaired(task, state) -> float:
    reward_hacking = ensure_reward_hacking_state(task, state)
    return 0.0 if reward_hacking is None else float(
        reward_hacking["simulator_contract_repaired"]
    )


class Tau2TasksetConfig(vf.TasksetConfig):
    domain: str = "telecom"
    user_model: str = DEFAULT_USER_MODEL
    user_args: vf.ConfigData | None = None
    user_base_url: str = DEFAULT_USER_BASE_URL
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR
    max_steps: int = DEFAULT_MAX_STEPS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_turns: int = DEFAULT_MAX_STEPS
    mutation_file: str | None = None
    mutation_ids: list[str] | None = None
    mutation_mode: str = "none"
    agent_elicitation_prompt: str | None = None
    agent_elicitation_position: str = "post_policy"


class Tau2Taskset(vf.Taskset):
    config_type = Tau2TasksetConfig

    def __init__(
        self,
        domain: str | None = None,
        *,
        user_model: str | None = None,
        user_args: ConfigMap | None = None,
        user_base_url: str | None = None,
        user_api_key_var: str | None = None,
        max_steps: int | None = None,
        max_errors: int | None = None,
        max_turns: int | None = None,
        mutation_file: str | None = None,
        mutation_ids: list[str] | None = None,
        mutation_mode: str | None = None,
        agent_elicitation_prompt: str | None = None,
        agent_elicitation_position: str | None = None,
        config: Tau2TasksetConfig | None = None,
    ):
        config = Tau2TasksetConfig(
            config,
            domain=domain,
            user_model=user_model,
            user_args=dict(user_args) if user_args is not None else None,
            user_base_url=user_base_url,
            user_api_key_var=user_api_key_var,
            max_steps=max_steps,
            max_errors=max_errors,
            max_turns=max_turns,
            mutation_file=mutation_file,
            mutation_ids=mutation_ids,
            mutation_mode=mutation_mode,
            agent_elicitation_prompt=agent_elicitation_prompt,
            agent_elicitation_position=agent_elicitation_position,
        )
        mutation_specs: list[MutationSpec] | None = None
        if config.mutation_mode not in {"none", "eval_only"}:
            raise ValueError(f"Unsupported mutation_mode: {config.mutation_mode!r}")
        if config.mutation_file is not None:
            mutation_path = resolve_mutation_path(
                config.mutation_file,
                base_dir=Path(__file__).parent,
            )
            mutation_specs = load_mutation_specs(
                mutation_path,
                domain=config.domain,
                mutation_ids=config.mutation_ids,
            )
            mutation_specs = validate_mutation_specs(
                mutation_specs,
                domain=config.domain,
            )
        resolved_user_args = (
            DEFAULT_LLM_ARGS_USER if config.user_args is None else config.user_args
        )
        session_factory = load_session_factory(
            domain=config.domain,
            user_model=config.user_model,
            user_args=tau2_user_args(
                resolved_user_args, config.user_base_url, config.user_api_key_var
            ),
            max_steps=config.max_steps,
            max_errors=config.max_errors,
        )
        toolset = load_toolset(
            domain=config.domain,
            user_model=config.user_model,
            user_args=resolved_user_args,
            user_base_url=config.user_base_url,
            user_api_key_var=config.user_api_key_var,
            max_steps=config.max_steps,
            max_errors=config.max_errors,
            session_factory=session_factory,
        )

        def load_rows():
            return source(
                config.domain,
                max_turns=config.max_turns,
                mutation_specs=mutation_specs,
                agent_elicitation_prompt=config.agent_elicitation_prompt,
                agent_elicitation_position=config.agent_elicitation_position,
            )

        super().__init__(
            source=load_rows,
            taskset_id=f"tau2_{config.domain}",
            setups=[make_tau2_setup(session_factory)],
            rewards=[tau2_reward],
            metrics=[
                tau2_num_errors,
                tau2_num_steps,
                tau2_num_assistant_tool_calls,
                tau2_num_user_tool_calls,
                tau2_buggy_reward,
                tau2_oracle_correct,
                tau2_exploit_detected,
                tau2_needs_manual_review,
                tau2_simulator_contract_repaired,
            ],
            toolsets=[toolset],
            user=vf.User(tau2_user, bindings={"session": session_factory}),
            config=config,
        )


def load_taskset(
    domain: str | None = None,
    *,
    user_model: str | None = None,
    user_args: ConfigMap | None = None,
    user_base_url: str | None = None,
    user_api_key_var: str | None = None,
    max_steps: int | None = None,
    max_errors: int | None = None,
    max_turns: int | None = None,
    mutation_file: str | None = None,
    mutation_ids: list[str] | None = None,
    mutation_mode: str | None = None,
    agent_elicitation_prompt: str | None = None,
    agent_elicitation_position: str | None = None,
    config: Tau2TasksetConfig | None = None,
) -> Tau2Taskset:
    return Tau2Taskset(
        domain=domain,
        user_model=user_model,
        user_args=user_args,
        user_base_url=user_base_url,
        user_api_key_var=user_api_key_var,
        max_steps=max_steps,
        max_errors=max_errors,
        max_turns=max_turns,
        mutation_file=mutation_file,
        mutation_ids=mutation_ids,
        mutation_mode=mutation_mode,
        agent_elicitation_prompt=agent_elicitation_prompt,
        agent_elicitation_position=agent_elicitation_position,
        config=config,
    )


def load_environment(
    domain: str = "telecom",
    *,
    user_model: str = DEFAULT_USER_MODEL,
    user_args: ConfigMap | None = None,
    user_base_url: str = DEFAULT_USER_BASE_URL,
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    max_turns: int = DEFAULT_MAX_STEPS,
    mutation_file: str | None = None,
    mutation_ids: list[str] | None = None,
    mutation_mode: str = "none",
    agent_elicitation_prompt: str | None = None,
    agent_elicitation_position: str = "post_policy",
    config: vf.EnvConfig,
) -> vf.Env:
    config = vf.EnvConfig(
        config,
        taskset=Tau2TasksetConfig(
            domain=domain,
            user_model=user_model,
            user_args=dict(user_args) if user_args is not None else None,
            user_base_url=user_base_url,
            user_api_key_var=user_api_key_var,
            max_steps=max_steps,
            max_errors=max_errors,
            max_turns=max_turns,
            mutation_file=mutation_file,
            mutation_ids=mutation_ids,
            mutation_mode=mutation_mode,
            agent_elicitation_prompt=agent_elicitation_prompt,
            agent_elicitation_position=agent_elicitation_position,
        ),
    )
    taskset_config = (
        None if config.taskset is None else Tau2TasksetConfig(config.taskset)
    )
    harness_config = (
        None if config.harness is None else vf.HarnessConfig(config.harness)
    )
    taskset = load_taskset(config=taskset_config)
    return vf.Env(taskset=taskset, harness=vf.Harness(config=harness_config))
