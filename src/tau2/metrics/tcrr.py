from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from loguru import logger

from tau2.data_model.message import Message, ToolMessage, AssistantMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.config import get_tcrr_window_size, get_tcrr_batch_threshold


@dataclass
class ToolCallInfo:
    name: str
    params: dict
    turn_idx: int
    assistant_turn_idx: int
    call_id: str
    correct: bool
    params_valid: bool


@dataclass
class TCRRResult:
    """
    Result of TCRR (Tool-Call Redundancy Ratio) computation.
    
    TCRR Formula:
    TCRR = (Window_Redundant_Calls + Batch_Redundant_Calls) / Total_Calls
    
    Where:
    - Window_Redundant_Calls: Number of calls that are identical to calls made 
      in the previous {window_size} assistant turns
    - Batch_Redundant_Calls: Number of identical calls within the same turn 
      that exceed the batch_threshold (only excess calls are counted)
    - Total_Calls: Total number of tool calls across all simulations
    
    TCRR-W (Window-based): Window_Redundant_Calls / Total_Calls
    TCRR-B (Batch): Batch_Redundant_Calls / Total_Calls
    """
    total_calls: int
    redundant_calls: int
    redundancy_ratio: float
    redundant_by_turn: Dict[int, int]
    window_size: int
    window_redundant_calls: int = 0
    batch_redundant_calls: int = 0


def normalized_params(params: dict) -> tuple:
    """Recursively normalize parameters for consistency."""
    normalized_items = []
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            normalized_items.append((k, normalized_params(v)))
        elif isinstance(v, list):
            try:
                normalized_list = []
                for item in v:
                    if isinstance(item, dict):
                        normalized_list.append(normalized_params(item))
                    elif isinstance(item, list):
                        normalized_list.append(tuple(str(subitem) for subitem in item))
                    else:
                        normalized_list.append(item)
                sorted_v = tuple(sorted(normalized_list))
            except TypeError:
                sorted_v = tuple(str(item) for item in v)
            normalized_items.append((k, sorted_v))
        else:
            normalized_items.append((k, v))
    return tuple(normalized_items)


def extract_tool_calls_with_turns(messages: List[Message]) -> List[ToolCallInfo]:
    """Extract tool calls with turn context information from all message types."""
    tool_calls = []

    tool_results = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.id] = {
                "success": not msg.error,
                "error": msg.error,
                "content": msg.content,
            }

    assistant_turn_count = 0

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            assistant_turn_count += 1
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                correct = True
                if hasattr(tool_call, "id") and tool_call.id in tool_results:
                    correct = tool_results[tool_call.id]["success"]
                params_valid = True
                if hasattr(tool_call, "id") and tool_call.id in tool_results:
                    result = tool_results[tool_call.id]
                    if result["error"] and result.get("content"):
                        error_content = str(result["content"]).lower()
                        if any(
                            keyword in error_content
                            for keyword in [
                                "invalid parameter",
                                "missing parameter",
                                "parameter error",
                                "bad parameter",
                                "invalid argument",
                                "missing argument",
                            ]
                        ):
                            params_valid = False

                tool_call_info = ToolCallInfo(
                    name=tool_call.name,
                    params=tool_call.arguments,
                    turn_idx=msg.turn_idx or 0,
                    assistant_turn_idx=assistant_turn_count,
                    call_id=getattr(tool_call, "id", ""),
                    correct=correct,
                    params_valid=params_valid,
                )
                tool_calls.append(tool_call_info)

    return tool_calls


def compute_tcrr_windowed(
    tool_calls: List[ToolCallInfo], window_size: int = 3, batch_threshold: int = 2
) -> TCRRResult:
    """
    Compute TCRR using window-based approach with batch redundancy detection.
    
    Args:
        tool_calls: List of tool calls with turn information
        window_size: Number of previous assistant turns to consider for redundancy
        batch_threshold: Maximum identical calls to same function+params in one turn 
                        before flagging excess calls as redundant

    Returns:
        TCRRResult with redundancy statistics
    """
    if not tool_calls:
        return TCRRResult(
            total_calls=0,
            redundant_calls=0,
            redundancy_ratio=0.0,
            redundant_by_turn={},
            window_size=window_size,
        )

    total_calls = len(tool_calls)
    window_redundant_calls = 0
    batch_redundant_calls = 0
    redundant_by_turn = {}

    calls_by_turn: Dict[int, List[ToolCallInfo]] = {}
    for call in tool_calls:
        turn = call.assistant_turn_idx
        if turn not in calls_by_turn:
            calls_by_turn[turn] = []
        calls_by_turn[turn].append(call)

    sorted_turns = sorted(calls_by_turn.keys())

    for current_turn in sorted_turns:
        current_calls = calls_by_turn[current_turn]
        redundant_by_turn[current_turn] = 0

        window_start = max(1, current_turn - window_size)
        previous_calls = []

        for prev_turn in range(window_start, current_turn):
            if prev_turn in calls_by_turn:
                previous_calls.extend(calls_by_turn[prev_turn])

        previous_identities: Set[Tuple[str, tuple]] = set()
        for prev_call in previous_calls:
            try:
                params_norm = normalized_params(prev_call.params)
                identity = (prev_call.name, params_norm)
                previous_identities.add(identity)
            except Exception as e:
                logger.warning(f"Error normalizing params for TCRR: {e}")
                identity = (prev_call.name, str(prev_call.params))
                previous_identities.add(identity)

        for call in current_calls:
            try:
                params_norm = normalized_params(call.params)
                identity = (call.name, params_norm)

                if identity in previous_identities:
                    window_redundant_calls += 1
                    redundant_by_turn[current_turn] += 1

            except Exception as e:
                logger.warning(f"Error normalizing params for TCRR: {e}")
                identity = (call.name, str(call.params))
                if identity in previous_identities:
                    window_redundant_calls += 1
                    redundant_by_turn[current_turn] += 1

        call_identities = {}
        for call in current_calls:
            try:
                identity = (call.name, normalized_params(call.params))
            except Exception:
                identity = (call.name, str(call.params))
            
            if identity not in call_identities:
                call_identities[identity] = 0
            call_identities[identity] += 1

        for identity, count in call_identities.items():
            if count > batch_threshold:
                excess_calls = count - batch_threshold
                batch_redundant_calls += excess_calls
                redundant_by_turn[current_turn] += excess_calls

    total_redundant_calls = window_redundant_calls + batch_redundant_calls
    redundancy_ratio = total_redundant_calls / total_calls if total_calls > 0 else 0.0

    return TCRRResult(
        total_calls=total_calls,
        redundant_calls=total_redundant_calls,
        redundancy_ratio=redundancy_ratio,
        redundant_by_turn=redundant_by_turn,
        window_size=window_size,
        window_redundant_calls=window_redundant_calls,
        batch_redundant_calls=batch_redundant_calls,
    )


def compute_tcrr(
    simulations: List[SimulationRun],
) -> Tuple[TCRRResult, Dict[str, TCRRResult]]:
    """
    Compute TCRR for each task separately and return both aggregated and per-task results.
    
    Args:
        simulations: List of simulation runs

    Returns:
        Tuple of (aggregated_result, results_by_task)
    """
    window_size = get_tcrr_window_size()
    batch_threshold = get_tcrr_batch_threshold()

    results_by_task = {}
    all_tool_calls = []

    # Group simulations by task
    sims_by_task: Dict[str, List[SimulationRun]] = {}
    for sim in simulations:
        task_id = sim.task_id
        if task_id not in sims_by_task:
            sims_by_task[task_id] = []
        sims_by_task[task_id].append(sim)

    # Compute TCRR for each task
    for task_id, task_sims in sims_by_task.items():
        task_tool_calls = []
        for sim in task_sims:
            tool_calls = extract_tool_calls_with_turns(sim.messages)
            task_tool_calls.extend(tool_calls)
            all_tool_calls.extend(tool_calls)
        results_by_task[task_id] = compute_tcrr_windowed(
            task_tool_calls, window_size, batch_threshold
        )

    aggregated_result = compute_tcrr_windowed(
        all_tool_calls, window_size, batch_threshold
    )

    return aggregated_result, results_by_task


def get_detailed_tcrr_breakdown(results_by_task: Dict[str, TCRRResult]) -> dict:
    """
    Get detailed TCRR breakdown for display purposes.
    """
    problematic_tasks = []
    for task_id, task_result in results_by_task.items():
        if task_result.redundancy_ratio > 0.3:
            problematic_tasks.append({
                'task_id': task_id,
                'redundancy_ratio': task_result.redundancy_ratio,
                'total_calls': task_result.total_calls,
                'redundant_calls': task_result.redundant_calls
            })
    
    problematic_tasks.sort(key=lambda x: x['redundancy_ratio'], reverse=True)
    
    return {
        'problematic_tasks': problematic_tasks[:5],
    }
